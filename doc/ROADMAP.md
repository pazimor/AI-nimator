# AI-nimator — Fiche de route canonique

> **Statut** : document de référence (canonical). Toute décision qui contredit
> ce document doit être tranchée par Pazimor, puis reportée ici.
>
> **Public** : agent d'implémentation (Sonnet) pour le code, agent
> d'expérimentation (Opus) pour le débogage du scaling, Pazimor pour les
> arbitrages.
>
> **Dernière consolidation** : 2026-06-11. Sources : repo (configs/code
> annotés), wiki Obsidian `projects/ai-nimator/` (9 pages v2, à jour
> 2026-05-12), sessions Claude (avril–juin 2026), résumé de la session
> "refactor & santé du modèle".

---

## 1. Le projet en une page

AI-nimator est un projet de recherche : générer des animations 3D
squelettiques (SMPL-22, compatible jeux vidéo) à partir de prompts texte,
entraîné sur AMASS (+ annotations HumanML3D/KIT). Pipeline : diffusion
conditionnelle (v-prediction, cosine schedule, sampling DDIM, CFG) avec
encodeur texte dédié et cross-attention text↔motion.

**Historique synthétique :**

| Génération | Période | Résultat |
|---|---|---|
| v1 (Étape 1) | 2025 → avril 2026 | 254M params, XLM-R frozen, 11 losses. `val_loss=0.43` (cible 0.32) après 318 epochs (~80h MPS). Animations exploitables uniquement en locomotion 120 frames. Abandonnée : 9 problèmes structurels qui se compensaient. |
| v2 (refonte) | mai 2026 | Stack 13× plus compact (~19M), lean representation (135 ch), v-pred + cosine, custom BPE encoder joint-trained. Phases A–F livrées, 323 tests verts. |
| v2 scaling | juin 2026 (en cours) | Le modèle conditionne correctement à petit N mais la qualité se dégrade quand N croît. Sweeps N=5→1000 et variantes de capacité (`n500_big`, `n1000_640`...) **non tranchés**. |

**Les deux objectifs de cette fiche :**

- **Goal A — Refactor "fiabiliser sans réécrire"** : monorepo propre,
  configs Pydantic, et surtout une surcouche santé/debug (HealthHub)
  intégrée au réseau. → délégué à **Sonnet** (sections 3–5).
- **Goal B — Déboguer le scaling** : comprendre pourquoi le modèle "tient"
  à 1 sample mais se dégrade vers N=1000, et scaler jusqu'à l'ensemble des
  dossiers préprocessés (~83k samples, gates intermédiaires). → piloté par
  **Pazimor + Opus**, outillé par Sonnet (section 6).

**Ordre d'exécution** (précisé 2026-06-12) : B1-bis (consolidation de la
recette — changement de config léger) passe en PREMIER. Ensuite A1
(configs résolues) et A3 (health/) avant tout nouveau gros run : un run
long sans `resolved_config.yaml` ni probes santé n'est plus acceptable —
c'est exactement ce qui a coûté le faux run "phaseE". A et B avancent
ensuite en parallèle, en évitant de toucher les mêmes fichiers. Aucune
réécriture big-bang : migration progressive, le training v2 actuel doit
rester exécutable à chaque étape.

---

## 2. Vérités canoniques (ne pas re-débattre)

Décisions actées, avec leur justification. Sonnet ne doit PAS les remettre
en cause ; si un blocage technique l'exige, escalader à Pazimor.

1. **v-prediction + cosine β schedule + DDIM (100 steps) + Min-SNR-γ=5**.
   Standard 2024+, bien conditionné aux extrêmes t=0/t=T. (wiki
   `v2-modern-diffusion`)
2. **Lean representation : 135 channels** = rotation6d (132) +
   root_translation (3). Tout signal FK-dérivable (joint_xyz, velocities,
   foot_contact, pelvis_height) est supervisé **à la loss** via FK, jamais
   prédit comme channel. (wiki `v2-lean-representation`)
3. **Z-normalization obligatoire et vérifiée**. Le bug le plus coûteux de
   la refonte était son oubli. Tout nouveau pipeline doit asserter
   post-norm ≈ N(0,1) et le checkpoint doit embarquer les stats du
   normalizer. (wiki `v2-z-normalization-bug`)
4. **Conditionnement multiplicatif non-bypassable** : cross-attention
   seule ⇒ collapse cond/uncond (`cfg_sim ≈ 0.9995` constaté Phase E).
   FiLM global + per-block AdaLN + null embedding learnable restent
   **activés par défaut**. (wiki `v2-cond-uncond-collapse`,
   `v2-phaseF-conditioning`)
5. **CFG dropout obligatoire** (`condMaskProb` 0.10–0.20) dès qu'on sample
   avec `cfgScale > 1`.
6. **Encodeur texte canonique : custom BPE 8k + transformer 4 layers 256d
   (~5M), output 384** (décision Pazimor 2026-06-11). Le frozen CLIP
   ViT-B/32 (`textEncoderType="clip"`) est conservé comme **outil de
   diagnostic/baseline**, pas comme cible produit. Conséquence refactor :
   les poids frozen ne doivent plus être sérialisés dans les checkpoints
   (cf. §5, les checkpoints CLIP font 1.35 GB vs ~80 MB).
   *(Amendement 2026-06-12 : le régime d'entraînement — joint vs
   pré-entraîné séparément puis gelé/fine-tuné — n'est plus acté ; il est
   tranché expérimentalement en étape B2, rendu possible par le
   découplage du point 9.)*
7. **Sanity check overfit-1-sample** : reste le smoke test de référence
   avant tout run long. Critères diagnose : `cfg_sim < 0.95`,
   `seed_sim < 0.90`, encoder cond↔uncond sim `< 0.5`.
8. **Conventions de code** : celles de `CLAUDE.md` (NumPy docstrings,
   Pylance strict, pas de magic numbers, méthodes ≤ 25 lignes, fichiers
   ~500 lignes, dataclasses dans `src/shared/types`, tests isolés).
9. **Encodeur texte découplé de la génération** (décision Pazimor
   2026-06-12). L'encodeur est un **module standalone** : interface
   stable (`TextEncoderProtocol` : `encode() → hidden states + mask +
   pooled`, `forwardNull()`), artefact propre (dossier
   `{config.yaml, tokenizer/, weights.pt}`), CLI d'entraînement dédié.
   La génération le **consomme comme une dépendance versionnée**
   (chemin d'artefact + flag `encoderTrainable` dans la config), elle ne
   le définit pas. Bénéfices : pré-entraînement séparé possible, swap
   custom↔CLIP sans toucher au denoiser, export indépendant. → Phase A7.
10. **Exportabilité ONNX préservée — ne jamais bloquer la voie NPU**
    (décision Pazimor 2026-06-12). Cible potentielle : Apple Neural
    Engine via ONNX Runtime (CoreML Execution Provider). Ce n'est pas un
    objectif de perf immédiat, c'est une **contrainte de conception** :
    encodeur et denoiser doivent rester exportables par
    `torch.onnx.export` à tout moment. Règles : pas de control flow
    dépendant des données dans les `forward()`, pas de `.item()` /
    listes Python sur le chemin du graphe, boucle DDIM **hors graphe**
    (le graphe exporté = un step de débruitage), axes dynamiques
    (batch, frames) déclarés. Un test d'export tourne en CI dès la
    phase A8 — toute PR qui le casse est refusée.

**Tension connue, à résoudre par le refactor (pas par débat)** : la
philosophie v2 était "3 losses", mais `full_training_v2.py` a ré-accumulé
~7 composants (diffusion, vel_xyz, jointPosition, footContact,
clipGuidance, auxPoolContrastive, x0Contrastive) + self-conditioning. Le
refactor ne supprime rien d'office : il rend chaque composant **visible,
activable par config, et observable** (poids effectif dans la loss
totale), pour que Goal B puisse trancher expérimentalement.

---

## 3. Goal A — Architecture cible du refactor

### 3.1 Constat

~22k lignes "vibe codées", éclatées historiquement en plusieurs projets
Poetry, avec du legacy v1 mort ou semi-mort (`train_generation.py` 1000+
lignes, feature `clip/`, profils `default`/`overfit` v1 de `network.yaml`,
caches texte XLM-R). Le debug d'incidents (ex. collapse Phase E) a été
douloureux faute d'indicateurs intégrés : les outils existent
(`pytorch-auditor`, skill `ai-debugging`, `diagnose_v2`) mais en "sac à
features mal organisé", hors du réseau.

### 3.2 Cible : monorepo en couches de dépendances (validé 2026-06-12)

Organisation **par couches** : chaque module n'importe que ce qui est
strictement en dessous de lui. La couche d'un fichier découle de ses
dépendances — fini le layout plat type `features/` + `shared/`.

```
src/ainimator/   # un seul package installable (imports: ainimator.*)
├── core/        # L0 — types (dataclasses), constantes, schéma config
│                #   Pydantic, device, logging. N'importe RIEN du projet.
├── geometry/    # L1 — quaternion, rot6d, FK, squelette SMPL-22.
│                #   Dépend de : core.
├── data/        # L2 — lecture AMASS .npz, link dataset, preprocess,
│                #   mirror, normalizer. Dépend de : core, geometry.
├── text/        # L3 — encodeur standalone (§2.9) : tokenizer BPE,
│                #   transformer, TextEncoderProtocol, wrapper CLIP.
│                #   Dépend de : core. (Exportable/swappable seul.)
├── diffusion/   # L3 — schedule cosine, v-pred, sampler DDIM (la boucle
│                #   vit ici, HORS graphe §2.10). Dépend de : core.
├── model/       # L3 — denoiser, FiLM/AdaLN, losses (FK via geometry).
│                #   Dépend de : core, geometry, text, diffusion.
│                #   ⛔ N'importe JAMAIS data.
├── health/      # L4 — Probe/Contract/HealthHub, évaluation, audit.
│                #   Dépend de : L0–L3.
├── training/    # L4 — boucles overfit/full, checkpoints, EMA.
│                #   SEUL module à voir data ET model (+ health).
├── export/      # L4 — ONNX, Collada/JSON, postprocess. Dépend de L0–L3.
└── cli/         # L5 — entrypoints SANS logique : config → pipeline.
configs/         # YAML versionnés (hors package)
legacy/          # v1 archivé (phase A5) — importable nulle part
test/            # miroir de src/ainimator/
```

Règles d'imports (les 4 qui comptent) :

1. **Imports uniquement vers le bas** — jamais latéraux entre L3 sauf
   `model → text/diffusion` (déclaré), jamais ascendants.
2. **`model` n'importe jamais `data`** : il travaille sur des tenseurs ;
   le contrat de batch est un type de `core`. (Testabilité + ONNX.)
3. **`training` est l'unique point de jonction** data ↔ model.
4. **`cli` est sans logique** : parse la config, appelle un pipeline.

**La règle est vérifiée, pas espérée** : contrats `import-linter`
(layers + interdits ci-dessus) exécutés en CI dès la phase A2 — tout
import qui remonte fait échouer le build. C'est l'équivalent structurel
du test d'export ONNX (§2.10).

Principes :

- **Configs YAML = feature flags.** Chaque option (loss, augmentation,
  conditioning, self-cond...) est déclarée dans un schéma Pydantic strict
  (`extra="forbid"`), avec défauts identiques au comportement actuel.
  **Chaque run sauvegarde sa config résolue** dans son `outputDir`
  (`resolved_config.yaml`) — fini les runs "phaseE" qui exécutaient en
  réalité l'archi de Phase D parce qu'un flag était resté à `False`.
- **Mode `--debug`** sur tous les CLI d'entraînement : modèle réduit,
  quelques dizaines de steps, probes santé à fréquence max, pour un cycle
  d'itération < 2 min.
- **Smoke test principal** : overfit sur un batch (déjà existant via le
  profil overfit) exécutable en une commande.

### 3.3 La surcouche santé : Probe / Contract / HealthHub

**Décision (2026-06-12)** : `health/` est **l'outil de debug global et
unique** du projet. Les outils actuels ne restent pas des scripts
séparés : ils sont **absorbés** comme des runtimes du HealthHub —
`diagnose_v2` (conditioning), l'adapteur `pytorch-auditor`
(`audit_adapter_v2.py`), `cross_prompt_sim_v2.py`, et les fonctions
d'évaluation des scripts de scaling (fidelity/retrieval/distinctness).
Un seul point d'entrée CLI : `poetry run python -m ainimator.cli.health
{watch|audit|diagnose|report}`.

Le graphe existe déjà (l'arbre `nn.Module`) : la surcouche s'y **attache
via les hooks PyTorch**, elle ne le redéfinit pas (pas de node editor).

Trois objets :

| Objet | Rôle | Contrainte clé |
|---|---|---|
| `Probe` | Hooks forward/backward sur un chemin de module ; capture des **stats scalaires** (mean/std/norm/rank...) | Ne stocke **jamais** de tenseurs ; coût borné |
| `Contract` | Critères déclaratifs **en YAML** par chemin de module ; produit des verdicts `OK / WARNING / CRITICAL` | Vocabulaire de verdicts aligné sur `pytorch-auditor` |
| `HealthHub` | Registre central : attache les probes, collecte, évalue les contracts, route vers TensorBoard + JSONL + alertes log | Deux runtimes, un seul hub |

Les runtimes (un seul hub, plusieurs modes) :

- `hub.step()` — pendant le training, échantillonné **1 step sur N**
  (configurable, défaut N=50 aligné sur `logEvery`). Publie métriques et
  verdicts en continu (TensorBoard + JSONL dans `outputDir/health/`).
- `hub.audit()` — hors training, sur un checkpoint : sert d'**adapteur
  niveaux 3–4 au `pytorch-auditor`** (implémente `build_dataloader` /
  `build_model` / `normalize` / `compute_loss` à partir des objets du
  projet — remplace `scripts/audit_adapter_v2.py`).
- `hub.diagnose()` — hors training, sur des générations : absorbe
  `diagnose_v2` et `cross_prompt_sim_v2` (cfg_sim, seed_sim, cross-prompt
  sim, sensibilité au conditionnement) + les métriques d'évaluation des
  scripts de scaling (fidelity / retrieval / distinctness sur le probe
  set standard walk/jump/sit/wave/run).
- `hub.report()` — agrège JSONL + verdicts d'un run en un rapport unique
  (markdown ou JSON) : la "fiche santé" qu'on lit après chaque run, avec
  pour chaque score sa valeur, sa cible et son verdict (cf. table §3.5).

Métriques ciblées sur les incidents passés (chaque probe doit répondre à
« quel incident passé ça aurait détecté ? ») :

| Douleur vécue | Métrique de détection | Incident de référence |
|---|---|---|
| Collapse cond/uncond | **effective rank** + cosine sim intra-batch des hidden states ; `cfg_sim` périodique | Phase E (cfg_sim ≈ 0.9995, découvert après 20h de run) |
| Prompt ignoré | **sensibilité au conditionnement** : forward avec conditioning réel vs shuffled, delta de loss | Run 212-epoch CLIP (cross-prompt sim 0.98) |
| Réseau sous-dimensionné / mort | ratio **‖Δw‖/‖w‖** par couche ; couches à gradient ~0 | adaln `cond_proj` à zéro (audit) ; rvel channels morts |
| Loss component mort ou dominant | part de chaque composant dans la loss totale (avec poids) | `loss_root_velocity=0.0000` silencieux en v1 |
| Pipeline data cassé | assert post-norm ≈ N(0,1) au premier batch ; NaN/Inf scan | bug z-normalization (refonte) |

**Garde-fou anti-usine à gaz** : démarrage avec **3 probes** (denoiser
blocks, text encoder pool, output head) et **5 contracts** (cond/uncond
sim, conditioning sensitivity, Δw/w, décomposition loss, post-norm N(0,1)).
Tout ajout ultérieur exige la justification "incident passé détecté".

### 3.5 Référence des scores : sens, cible, seuils

Table canonique. Ces seuils sont **encodés dans `src/configs/health.yaml`**
(un commentaire YAML par métrique reprend la ligne correspondante), et
`hub.report()` affiche pour chaque score : valeur, direction (↑/↓), cible,
verdict. Les seuils sont des points de départ calibrés sur les incidents
passés — les recalibrer est permis, mais chaque changement se documente ici.

| Métrique | Mesure | Direction | Sain | WARNING | CRITICAL | Incident de référence |
|---|---|---|---|---|---|---|
| `cfg_sim` | cosine génération cond vs uncond (même seed) | ↓ vers 0 | < 0.95 | 0.95–0.99 | > 0.99 | Phase E : 0.9995, prompt ignoré |
| `seed_sim` | cosine entre générations de seeds ≠ (même prompt) | ↓ (diversité) | < 0.90 | 0.90–0.97 | > 0.97 | Phase E : 0.97, mode collapse |
| `encoder_cond_uncond_sim` | cosine pool encoder cond vs null | ↓ vers 0 | < 0.50 | 0.50–0.75 | > 0.75 | BOS-collapse : 0.7361 |
| `cross_prompt_sim` | cosine générations de prompts ≠ | ↓ | < 0.80 | 0.80–0.95 | > 0.95 | Run 212-epoch : 0.98 |
| `fidelity` | FK-cosine(gen, GT) sur le probe set | ↑ vers 1 | > 0.60 | 0.20–0.60 | < 0.20 (≈0 = aléatoire, <0 = anti-corrélé) | Sweep N=100 : −0.151 |
| `retrieval` | fraction où nearest-GT(gen_i) == GT_i | ↑ vers 1 | ≥ 0.80 | 0.40–0.80 | ≤ chance (1/n_probes, soit 0.20 à 5 probes) | Sweep N=100 : 0.20 |
| `distinctness` | FK-cosine moyen inter-générations | ↓ vers 0 | < 0.50 | 0.50–0.80 | > 0.80 | Sweep N=100 : +0.394 (dérive) |
| `conditioning_sensitivity` | Δloss forward texte réel vs shuffled | ↑ > 0 | nettement > 0 et stable | ≈ 0 sur 1 fenêtre | ≈ 0 persistant (prompt sans effet) | Collapse Phase E (détection tardive) |
| `effective_rank` | rank effectif des hidden states (par bloc) | ↑ (proche de min(B·F, D) utile) | > 0.5 × D | 0.1–0.5 × D | < 0.1 × D | Collapse : rank ≈ 1 |
| `intra_batch_sim` | cosine moyen des hidden states intra-batch | ↓ | < 0.5 | 0.5–0.9 | > 0.9 (états identiques) | Collapse Phase E |
| `update_ratio` | ‖Δw‖/‖w‖ par couche et par step | plage | 1e-4 – 1e-2 | < 1e-5 (couche gelée) ou > 1e-1 | ≈ 0 persistant ou NaN | `cond_proj` AdaLN à zéro |
| `loss_share` | part pondérée de chaque composant dans la loss totale | équilibre | chaque composant 1–70 % | composant < 1 % (mort) ou > 80 % (dominant) | composant exactement 0.0 | `loss_root_velocity=0.0000` v1 |
| `post_norm_stats` | mean/std par channel après z-norm (1er batch) | mean→0, std→1 | \|mean\| < 0.1 et \|std−1\| < 0.1 | < 0.3 | au-delà (z-norm cassée/oubliée) | Bug z-normalization v2 |
| `nan_inf` | scan NaN/Inf poids + activations + gradients | 0 | 0 | — | ≥ 1 | — |
| `val_gap` | (val − train)/train sur loss diffusion | ↓ | < 15 % | 15–40 % | > 40 % (mémorisation) | rtrans gap +55 % epoch 144 v1 |
| `epoch_time` / `rss_memory` | temps/epoch et mémoire résidente | stable | dérive < 1.5× depuis epoch 1 | 1.5–3× | > 3× (swap/cache LRU saturé) | Epochs 1.3k s → 128k s (mai 2026) |

Notes d'interprétation pour les non-évidents :

- **fidelity ≈ 0 n'est pas "moyen", c'est aléatoire** : une FK-cosine se
  dégrade vers 0 quand la génération n'a plus rien à voir avec la GT, et
  devient négative si elle est anti-corrélée. Le succès se joue donc
  entre 0.6 et 1, pas entre 0 et 1 uniformément.
- **retrieval se lit contre la chance** : avec 5 probes, 0.20 = hasard
  pur. Le seuil sain (0.80) signifie "4 générations sur 5 sont plus
  proches de leur propre GT que de toute autre".
- **distinctness bas n'est bon que si fidelity est haut** : 5 générations
  très distinctes mais toutes fausses restent un échec. Lire les trois
  métriques ensemble (c'est le rôle de `hub.report()`).
- **cfg_sim et seed_sim se dégradent silencieusement** : ce sont les deux
  scores à surveiller en continu pendant le training (contract en
  `hub.step()`), pas seulement en fin de run — le collapse Phase E a
  coûté ~20h de run avant d'être vu.
- **fidelity / retrieval / distinctness s'évaluent à cfg ∈ {1, 4, 6}**
  (acquis 2026-06-12, run `n1000_640`) : un modèle sain à l'échelle peut
  donner retrieval 0.20 à cfg=1 et 1.00 à cfg=6 — le conditionnement est
  contenu dans le modèle, le CFG l'exprime. Les seuils de la table
  s'appliquent à la **meilleure** ligne cfg ; un retrieval faible à TOUS
  les cfg est le vrai signal d'échec. `hub.diagnose()` produit les trois
  lignes.

### 3.4 Ce qu'on garde / ce qu'on archive

| Sort | Éléments |
|---|---|
| **Garder (cœur v2)** | `full_training_v2`, `training_v2` (overfit), `denoiser_v2`, `sampler_v2`, `losses_v2`, `postprocess_v2`, custom tokenizer/encoder, `mirror_v2`, CLI `*_v2`, `preprocessed_dataset`, skeleton/FK, checkpoint_io |
| **Absorber dans `health/`** (puis archiver les originaux en fin de A3) | `diagnose_v2` + CLI `diagnose_generation_v2` → `hub.diagnose()` ; `scripts/audit_adapter_v2.py` → `hub.audit()` ; `scripts/cross_prompt_sim_v2.py` → `hub.diagnose()` ; fonctions `_evaluate`/`_fk`/`_cos`/`_pickProbes` de `scripts/scaling_curve_v2.py` → module d'évaluation `health/evaluation.py` (les scripts de sweep deviennent de simples orchestrateurs qui appellent health/) |
| **Archiver** (déplacer vers `legacy/`, exclu du package, importable nulle part) | `train_generation.py` v1, `features/clip/` (CLIP contrastif v1), `generate_animation.py` v1, `diagnostics.py` v1, `generation_text_cache.py` + `precompute_generation_text_cache.py` (cache XLM-R), profils `default` et `overfit` v1 de `network.yaml`, `train_clip` CLI |
| **Supprimer** | `__pycache__` versionnés, caches morts, code mort détecté par vulture/coverage après migration |

L'archivage est **réversible** (git) et ne se fait qu'à la phase A5, une
fois la parité v2 démontrée.

---

## 4. Goal A — Plan d'exécution pour Sonnet

Chaque phase = un incrément livrable, testé, sans casser
`train_generation_v2 --profile {overfit,full}`. Critères d'acceptation
explicites ; ne pas passer à la phase suivante si l'un d'eux échoue.

### Phase A1 — Schéma de config Pydantic + config résolue
- Créer `src/shared/config_schema.py` (ou `src/configs/schema.py`) :
  modèles Pydantic stricts couvrant `V2FullTrainingConfig`,
  `MotionDenoiserV2Config`, encoder, diffusion, losses. Les YAML existants
  (`network.yaml` profil v2, `text_encoder.yaml`, `dataset.yaml`) doivent
  se charger tels quels.
- Tout run écrit `resolved_config.yaml` (config complète, défauts inclus,
  + git SHA + date) dans `outputDir`.
- **Acceptation** : un YAML avec une clé inconnue ou un type faux lève une
  erreur de validation listant le champ fautif ; un run overfit court
  produit le fichier résolu ; tests unitaires du schéma.

### Phase A2 — Réorganisation en couches ✅ DONE (2026-06-12)
- Déplacements purs vers la structure §3.2 (git mv + mise à jour
  imports). Pas de refactor de logique dans cette phase. `test/` suit le
  même arbre (`test/ainimator/`). Le package est `src/ainimator/` ; les
  commandes passent de `python -m src.cli.*` à
  `python -m ainimator.cli.*` — CLAUDE.md et ce document mis à jour.
- `import-linter` configuré (`.importlinter`) avec 2 contrats actifs :
  layers (L0→L5, imports ascendants interdits) et `model ✗→ data`.
  Contrat `* ✗→ legacy` préparé en commentaire pour phase A5.
  Commande : `poetry run lint-imports`.
- **Résultat** : 431 tests verts (+ 1 pré-existant `test_root_translation_zeroing`),
  CLI v2 fonctionnels sous `python -m ainimator.cli.*`,
  `lint-imports` vert (2/2 contrats), test négatif documenté dans ce
  commit : introduire `from ainimator.model.denoiser_v2 import ...` dans
  `ainimator/core/` casse le lint avec
  `ainimator.core.* -> ainimator.model.*` BROKEN.
- Note d'architecture : `sampler_v2` placé dans `model/` (uses denoiser,
  above diffusion in hierarchy) ; `clip/data.py` (v1) placé dans
  `training/` (consomme à la fois model et data) ; ces deux fichiers
  iront dans `legacy/` en A5.

### Phase A3 — `health/` : l'outil de debug global (★ priorité)
- Implémenter Probe / Contract / HealthHub (§3.3) + intégration dans
  `full_training_v2` et `training_v2` derrière un flag config
  (`health.enabled`, défaut `true`, `health.everySteps`, défaut 50).
- Les 3 probes + 5 contracts initiaux, définis dans
  `src/configs/health.yaml` — **chaque métrique y porte en commentaire sa
  ligne de la table §3.5** (direction, cible, seuils).
- Les 4 runtimes : `step()`, `audit()` (adapteur pytorch-auditor 3–4),
  `diagnose()` (absorbe diagnose_v2 + cross_prompt_sim + évaluation
  fidelity/retrieval/distinctness), `report()` (fiche santé agrégée d'un
  run : valeur + direction + cible + verdict par score).
- CLI unique `ainimator/cli/health.py` : `watch` (suivre un run en cours
  via ses JSONL), `audit <checkpoint>`, `diagnose <checkpoint>`,
  `report <runDir>`.
- Sorties : JSONL par run (`outputDir/health/*.jsonl`) + TensorBoard +
  verdicts WARNING/CRITICAL dans le log d'entraînement.
- En fin de phase : archiver les originaux absorbés (`diagnose_v2`,
  `audit_adapter_v2`, `cross_prompt_sim_v2`) après vérification de parité
  des sorties sur un même checkpoint.
- **Acceptation** : sur un run overfit, le JSONL contient les 5 contracts
  évalués ; `health report` produit la fiche santé avec les 16 métriques
  de la table §3.5 (valeur, cible, verdict) ; parité numérique
  `hub.diagnose()` vs ancien `diagnose_generation_v2` sur un même
  checkpoint (tolérance 1e-4) ; un test injecte un collapse artificiel
  (hidden states constants) et vérifie le verdict CRITICAL ; un test
  injecte un texte shuffled sans effet sur la loss et vérifie le WARNING
  de conditioning ; surcoût wall-clock mesuré < 5 % à `everySteps=50`.

### Phase A4 — Checkpoints légers + hygiène des runs
- Exclure les poids frozen (CLIP) de la sérialisation : sauvegarder la
  référence (`clipModelName`) et les seuls modules entraînables.
  Rétro-compatibilité : `loadCheckpointV2` doit charger les anciens
  checkpoints 1.35 GB ET les nouveaux.
- Standardiser le layout d'un run : `outputDir/{checkpoints,health,
  resolved_config.yaml,log.txt}`.
- **Acceptation** : checkpoint custom-BPE ~taille des poids entraînables
  (ordre 80–150 MB) ; round-trip save/load testé ; ancien checkpoint
  chargeable.

### Phase A5 — Archivage du legacy v1
- Déplacements §3.4 vers `legacy/`, nettoyage des configs (supprimer les
  profils v1 de `network.yaml` ou les déplacer dans `legacy/configs/`).
- **Acceptation** : aucun import de `legacy/` dans `src/` ;
  tests verts ; `train_generation_v2` et `generate_animation_v2`
  inchangés ; CLAUDE.md mis à jour (commandes v1 retirées).

### Phase A6 — Mode `--debug` + smoke tests
- Flag `--debug` sur le CLI training : modèle réduit (config dédiée),
  ~50 steps, health à chaque step, durée cible < 2 min sur MPS.
- Cible make/poetry unique pour le smoke test overfit-1-batch.
- **Acceptation** : `--debug` tourne de bout en bout (training → checkpoint
  → 1 génération) en < 2 min ; documenté dans le README/CLAUDE.md.

### Phase A7 — Encodeur texte standalone (découplage, §2.9)
- Consolider le module `text/` (couche L3, §3.2) : tokenizer BPE +
  transformer + `TextEncoderProtocol` (interface que custom ET wrapper
  CLIP implémentent — le swap se fait par config, zéro changement de
  code). Dépendance unique : `core` — vérifiée par import-linter.
- Format d'artefact encodeur : dossier `{config.yaml, tokenizer/,
  weights.pt}` + hash. La config génération référence un **chemin
  d'artefact** + flag `encoderTrainable` (True = fine-tune joint comme
  aujourd'hui, False = gelé).
- CLI `train_text_encoder` : entraînement/pré-entraînement de l'encodeur
  seul (l'objectif de pré-entraînement — contrastif texte↔motion
  TMR-style ou autre — est défini par l'étape B2, le CLI fournit la
  mécanique).
- Checkpoints génération : ne sérialisent plus l'encodeur gelé
  (référence + hash seulement) ; sérialisent les poids encodeur
  uniquement si `encoderTrainable=True`.
- **Acceptation** : un run génération charge l'encodeur depuis un
  artefact ; swap custom↔CLIP par config seul ; parité numérique avec le
  comportement joint actuel quand `encoderTrainable=True` (mêmes losses
  sur run déterministe court) ; round-trip artefact save/load testé.

### Phase A8 — Exportabilité ONNX (voie NPU, §2.10)
- `export/onnx.py` + CLI `export_onnx {encoder|denoiser}` : exporte
  l'encodeur complet et **un step** de débruitage du denoiser (la boucle
  DDIM reste en Python/torch — elle orchestre le graphe, elle n'y vit
  pas). Axes dynamiques : batch, frames, longueur texte.
- Audit du code existant contre les règles §2.10 (control flow
  data-dependent, `.item()`, masks) ; corriger ce qui bloque l'export
  sans changer la sémantique (tests de non-régression).
- Test pytest de parité : sortie ONNXRuntime (CPU) vs torch sur les
  mêmes entrées, tolérance 1e-3 ; marqué pour tourner en CI.
- Hors scope (plus tard, si besoin réel de perf) : quantization, CoreML
  EP réel sur ANE, export du sampler complet.
- **Acceptation** : `export_onnx` produit des `.onnx` valides
  (onnx.checker) pour encodeur + denoiser-step ; test de parité vert ;
  liste des ops/patterns interdits documentée dans `export/README.md`.

**Règles pour Sonnet** : respecter les conventions §2.8 ; chaque phase =
commits atomiques ; ne jamais modifier les valeurs par défaut des
hyperparamètres existants sans instruction explicite ; en cas d'ambiguïté,
poser la question plutôt que de choisir silencieusement.

**Précision sur la règle "pas de run > 30 min"** : les runs explicitement
listés dans l'étape courante du protocole B (★) sont pré-autorisés par
Pazimor — la règle des 30 min s'applique à tout run NON listé. En
pratique, les agents **préparent** la commande (script + config + durée
estimée) et c'est Pazimor qui la lance dans son terminal : un entraînement
de plusieurs heures ne doit pas vivre à l'intérieur d'une session d'agent.

---

## 5. Goal B — Déboguer le scaling (état des lieux)

### 5.1 Le problème — RÉSOLU le 2026-06-12

Le modèle v2 (~19M, 4×384d) réussissait l'overfit 1-sample mais "ignorait
le prompt" quand N croissait. **Cause racine identifiée (run
`n1000_640`)** : denoiser sous-dimensionné ET sampling sans CFG. Les deux
corrigés, N=1000 est entièrement résolu (retrieval 5/5 à cfg=6).
Objectif final inchangé : entraîner sur **tous les dossiers préprocessés
(~83k samples)** — avec un doute assumé sur la nécessité réelle de tout
AMASS (à trancher par les gates ci-dessous, pas par avance).

### 5.2 Acquis expérimentaux (juin 2026)

- **Sweep à budget de steps constant** (`scaling_curve_v2.py`, N=5/20/100/
  400, ACCAD, frozen CLIP comme encodeur de diagnostic) : la fidelity
  s'effondre à N=100 (−0.151, retrieval 0.20) — mais l'exposure/sample
  passait de 700 à 56. **Confondu : sous-entraînement vs plafond de
  capacité.**
- **Test décisif N=100 @ ~600 exposures** (`decisive_n100_v2.py`) : la
  fidelity remonte → le cliff N=100 était du sous-entraînement.
- **Loi capacité↔N établie sur 3 points propres** (runs juin 7–12,
  consolidés dans LOG.md) :

  | Denoiser | Params | Tient (retrieval ~1.0 avec CFG) |
  |---|---|---|
  | 384d / 4L | ~17M | N ≤ ~50 |
  | 512d / 6L | ~45M | N ≤ ~500 |
  | 640d / 8L | ~90M | N = 1000 ✓ (5/5 à cfg=6) |

  Soit **params ≈ ∝ N** (~doubler les params quand N double).
  ⚠ Loi mesurée en **régime mémorisation** (train==val, retrieval des
  samples d'entraînement) ; on ne sait pas encore si la généralisation
  infléchit la courbe à grand N (c'est l'enjeu de B1-ter/B3).
- **Recette validée (2026-06-12)** : capacité ∝ N ; `cond-mask-prob 0.10`
  (sinon collapse dégénéré + CFG inutilisable) ; **CFG 4–6 à
  l'inférence** — le modèle contient le conditionnement mais a besoin du
  CFG pour l'exprimer (`n1000_640` : cfg=1 → retrieval 0.20, cfg=4 →
  0.80, cfg=6 → 1.00 avec fidelity +0.61 et distinctness +0.034) ;
  exposition suffisante (~600 exp/sample, grad-accum 1, pas de cap).
- Conséquence pour `health/` (§3.5) : fidelity / retrieval / distinctness
  s'évaluent **à cfg ∈ {1, 4, 6}**, pas seulement à cfg=1 — un retrieval
  faible à cfg=1 avec un retrieval fort à cfg=6 est un comportement
  SAIN, pas un échec. `hub.diagnose()` doit produire les trois lignes.
- Métriques du protocole : **fidelity** (FK-cosine gen vs GT),
  **retrieval** (nearest-GT == soi), **distinctness** (FK-cosine
  inter-gen, bas = mieux), + critères diagnose (`cfg_sim`, `seed_sim`,
  cross-prompt sim).

### 5.3 Protocole de reprise (Opus + Pazimor)

Étape B0 — **Consolider les runs de juin.** ✅ FAIT (2026-06-12) :
résultats consolidés dans `doc/experiments/LOG.md` (sweep N, decisive
N=100, échelle de capacité, run final `n1000_640`). La request 001 reste
ouverte uniquement pour archiver les valeurs brutes par run si les
fichiers `/tmp` existent encore.

Étape B1 — **Séparer données vs capacité.** ✅ TRANCHÉ (2026-06-12) :
la capacité était le facteur limitant, ET il manquait le CFG à
l'inférence. Loi établie (cf. §5.2) : params ≈ ∝ N, recette validée à
N=1000 avec 640d/8L. Plus rien à débattre à cette échelle.

Étape B1-bis — **Consolider la recette** — ★ ÉTAPE COURANTE (choix
Pazimor 2026-06-12, sur recommandation d'Opus ; B1-ter et B1-quater
viennent après) :
- Figer la recette dans le profil v2 de `network.yaml` (640d/8L,
  `cond-mask-prob 0.10`, exposition sans cap) + commande d'entraînement
  "propre" documentée — c'est un changement de défauts validé par
  Pazimor via cette fiche, l'implementer peut l'exécuter.
- **Garde-fou encodeur** : la consolidation reste sur le frozen CLIP
  (continuité avec la loi, qui a été établie avec lui). Ne PAS figer
  `textEncoderType="clip"` comme défaut canonique — §2.6 (custom BPE)
  reste la cible, revalidée en B2. La loi capacité↔N devra être
  re-vérifiée sur 1 point lors du passage au custom BPE.
- Traiter la **qualité** maintenant que le conditionnement est acquis :
  jitter → `--smooth-sigma`, vitesses irrégulières → renforcer la loss
  vélocité. Objectif : premier modèle réellement utilisable à N=1000.
- **Élargir le probe set** : 5 probes ⇒ retrieval 5/5 est un signal
  statistiquement faible. Passer à ≥ 20 probes (couvrant les verbes du
  probe set + variations de formulation) avant toute conclusion B1-ter ;
  ajouter des **prompts held-out** (jamais vus au training) pour obtenir
  une première mesure de généralisation — absente du protocole jusqu'ici.
- **Traçabilité intérimaire** (tant que A1 n'est pas livré) : chaque run
  copie manuellement sa commande complète + le diff de config dans
  `<outputDir>/run_command.txt`. Aucun run "mystère" de plus.

Étape B1-ter — **Extrapoler la loi avant la vraie cible** : un point
N=2000 (nécessite d'ajouter un folder, ex. CMU) en 640/8 ET en 768/10.
Question clé : la loi params ∝ N (mémorisation) tient-elle, ou la
généralisation l'infléchit-elle quand les motions se recouvrent ? C'est
LE point qui décide de la faisabilité de 56k/83k (cf. §5.4).

Étape B1-quater (optionnelle) — **Réduire la dépendance au CFG** :
ré-entraîner N=1000/640-8 avec `--min-snr-gamma 0` pour voir si le
retrieval à cfg=1 remonte (entraîner le haut-bruit). Utile mais non
bloquant : CFG 4–6 est une solution acceptable.

Étape B2 — **Re-basculer sur l'encodeur custom BPE** (canonique, §2.6) au
niveau N validé en B1, comparer au frozen CLIP à config égale. Critère :
écart fidelity/retrieval < 10 % ⇒ custom confirmé ; sinon documenter et
remonter à Pazimor (le custom reste la cible, mais le gap devient un item
de travail explicite : curriculum, vocab, profondeur encoder).
Une fois le découplage A7 livré, B2 inclut la comparaison des **régimes
d'entraînement de l'encodeur** (cf. amendement §2.6) à config égale :
(a) joint comme aujourd'hui, (b) pré-entraîné contrastif texte↔motion
puis gelé, (c) pré-entraîné puis fine-tuné. Verdict consigné dans
LOG.md et reporté en §2.

Étape B3 — **Palier 5 folders (~56k)** : full run avec la config gagnante
de B1/B2, `maxSamplesPerEpoch` et early-stopping recalibrés, EMA réactivée
(`emaDecay=0.9999`) pour les runs longs. Surveillance via health/ ;
critères de sortie : diagnose OK (§2.7) + validation visuelle Blender sur
le probe set standard (walk/jump/sit/wave/run).

Étape B4 — **Gate "tout AMASS" (~83k)** : n'ajouter les 12 folders
restants (Eyes_Japan 13k, MPI_HDM05 6.6k, etc.) **que si** B3 montre que
le modèle est data-bound (val loss continue de descendre, pas de plateau
de capacité) ET que les folders ajoutés couvrent des prompts manquants.
Sinon, ~56k est le point de fonctionnement et l'effort va à la qualité.

### 5.4 Contraintes matérielles

**Implication directe de la loi capacité↔N** : l'extrapolation naïve
(params ∝ N, régime mémorisation) donne ~180M pour N=2000, ~360M pour
N=4000... et un ordre de grandeur intenable (>1B) bien avant 56k. Trois
issues possibles, departagées par B1-ter : (a) la généralisation infléchit
la courbe (les motions distinctes se recouvrent, le modèle factorise au
lieu de mémoriser — c'est le pari standard du deep learning), (b) un
plafond de capacité local acceptable (~200–300M max sur MPS 32 GB) avec
un dataset curé plutôt qu'exhaustif, (c) compute distant (décision achat,
Pazimor). Ne PAS lancer de run 56k avant le verdict B1-ter.

Entraînement local : Mac 32 GB, MPS, batch 2–8 + gradient accumulation.
Points de vigilance documentés : epochs qui passent de ~1.3k s à ~128k s
(swap RAM / saturation du cache LRU / croissance du dataset — diagnostic
session mai 2026) → le health/ doit logger RSS mémoire et temps/step pour
détecter la dérive tôt. Si B1 conclut qu'il faut significativement plus de
capacité, prévoir l'option machine distante (déjà évoquée : "spark") comme
décision séparée.

### 5.5 Questions ouvertes (ne pas trancher sans données)

- `x0ContrastiveWeight` et `useSelfConditioning` : gardés en config,
  désactivés par défaut ; à évaluer un par un en B1 si le conditioning
  régresse au scale.
- Tout-AMASS nécessaire ? → gate B4.
- Retour éventuel du foot-skating loss en fine-tuning final uniquement.

---

## 6. Répartition des rôles

| Acteur | Périmètre | Interdits |
|---|---|---|
| **Sonnet (implementer)** | Goal A phases A1→A8, dans l'ordre, critères d'acceptation obligatoires ; outillage demandé par B (scripts, matrice d'expériences) ; traite les **demandes d'extraction de données** de l'experimenter (`doc/experiments/requests/`) | Changer les défauts d'hyperparamètres ; toucher aux décisions §2 ; lancer des runs > 30 min sans demande |
| **Opus (experimenter)** | Goal B : analyse des runs, hypothèses, protocole B0→B4 ; quand une donnée/un outil manque, il **n'implémente pas** : il dépose une demande dans `doc/experiments/requests/` (spec : quoi, format, checkpoints sources, critère de done) à dispatcher vers l'implementer | Refactor structurel hors health/ (le signaler à Sonnet via cette fiche) |
| **Sonnet (reviewer)** | Lecture seule : vérifie les critères d'acceptation d'une phase + conventions CLAUDE.md avant de la déclarer terminée | Modifier des fichiers |
| **Pazimor** | Arbitrages (§2, gates B1/B2/B4, achat compute), validation visuelle Blender, mise à jour de cette fiche | — |

**Maintenance de la fiche** : toute expérience conclue ajoute une ligne
dans `doc/experiments/LOG.md` (date, run, config, métriques, verdict) ;
toute décision nouvelle modifie la section concernée ici, avec date.

---

## 7. Références

- Wiki Obsidian (`/Users/pazimor/wiki-ia/projet-IA/projects/ai-nimator/`) :
  `v2-refonte-overview`, `v2-custom-text-encoder`, `v2-compact-denoiser`,
  `v2-lean-representation`, `v2-modern-diffusion`,
  `v2-z-normalization-bug`, `v2-multi-sample-training`,
  `v2-cond-uncond-collapse`, `v2-phaseF-conditioning`.
- Code richement annoté (les commentaires datés font foi d'historique) :
  `src/ainimator/training/full_training_v2.py`,
  `src/configs/network.yaml` (profil v2),
  `src/configs/train_generation.yaml`.
- Scripts d'expérimentation : `scripts/scaling_curve_v2.py`,
  `scripts/decisive_n100_v2.py`, `scripts/cross_prompt_sim_v2.py`.
- Outils : skill `pytorch-auditor` (audit checkpoint), skill
  `ai-debugging` (méthodologie),
  `src/ainimator/cli/diagnose_generation_v2.py`
  (absorbé par `ainimator/cli/health.py` à l'issue de la phase A3
  — cf. §3.3).
