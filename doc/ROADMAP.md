# AI-nimator — Fiche de route canonique (cadrage projet)

> **Statut** : document de référence (canonical). Il ne contient plus de
> plan d'exécution : il garde uniquement **ce qui cadre le projet** — les
> vérités canoniques, l'architecture en couches, la surcouche santé et la
> répartition des rôles. L'exécution vit dans les fiches dédiées.
>
> **Pivot 2026-06-23 — projet full-déterministe.** La voie diffusion
> est **abandonnée**. Le
> moteur unique est le **contrôleur autorégressif déterministe**, augmenté
> d'un **encodeur texte intégré** qui lui permet de recevoir des prompts
> (modèle « tout-en-un »). Voir :
> - `doc/ROADMAP_DETERMINIST.md` — **Goal A**, le moteur (contrôleur +
>   encodeur ; phases **A0→A7**).
> - `doc/ROADMAP_PLUGINS.md` — **Goal B**, le produit (plugins Unity /
>   Unreal ; phases **B0→B6**).
>
> Toute décision qui contredit ce document doit être tranchée par Pazimor,
> puis reportée ici avec une date.

---

## 1. Le projet en une page

AI-nimator est un projet de recherche : générer des animations 3D
squelettiques (SMPL-22, compatible jeux vidéo) à partir de **prompts texte**
et/ou de **signaux de contrôle**, entraîné sur AMASS (+ annotations
HumanML3D/KIT).

Depuis le pivot du 2026-06-23, le pipeline est **entièrement déterministe** :
un contrôleur autorégressif régresse le mouvement **une frame à la fois**
(`f(state_t, control_t, [prompt], [phase]) → Δstate`), sans boucle de
débruitage. Un encodeur texte intégré transforme un prompt en
conditionnement, ce qui rend le modèle « tout-en-un » : il peut être piloté
par un vecteur de contrôle (temps réel, jeu) **et/ou** par du texte.

**Historique synthétique :**

| Génération | Période | Résultat |
|---|---|---|
| v1 | 2025 → avril 2026 | 254M params, XLM-R frozen, 11 losses. Abandonnée (9 problèmes structurels). |
| v2 diffusion | mai–juin 2026 | Stack compact (~19M), lean repr (135 ch), v-pred + cosine, custom BPE. Loi capacité↔N établie. **Voie abandonnée au pivot 2026-06-23.** |
| déterministe | juin 2026 → | Contrôleur autorégressif + encodeur texte intégré. Seule voie active. Goal A (phases A0→A7, moteur) + Goal B (phases B0→B6, plugins). |

Les acquis de la voie diffusion qui **cadrent encore le projet**
(représentation lean, z-normalisation, FK à la loss, architecture en
couches, surcouche santé, encodeur texte découplé, exportabilité ONNX) sont
conservés ci-dessous. Les acquis purement diffusion (schedule, sampling,
CFG) sont marqués comme historiques.

---

## 2. Vérités canoniques (ne pas re-débattre)

Décisions actées, avec leur justification.

> **Marquage post-pivot (2026-06-23)** : les vérités **#1** (v-pred / cosine
> / DDIM / Min-SNR) et **#5** (CFG dropout) étaient propres à la diffusion,
> désormais abandonnée — elles restent listées comme **acquis historiques**,
> inactives. Toutes les autres **cadrent le projet et restent actives** ;
> #4 et #6 sont **réinterprétées** pour le déterministe (conditionnement par
> contrôle + texte au lieu de texte seul).

1. *(historique, diffusion)* **v-prediction + cosine β schedule + DDIM (100
   steps) + Min-SNR-γ=5**. Standard 2024+, bien conditionné aux extrêmes
   t=0/t=T.
2. **Lean representation : 135 channels** = rotation6d (132) +
   root_translation (3). Tout signal FK-dérivable (joint_xyz, velocities,
   foot_contact, pelvis_height) est supervisé **à la loss** via FK, jamais
   prédit comme channel. *(Socle hérité de la diffusion ; l'état **régressé**
   par le contrôleur fait **136** = rot6d 132 + root motion local
   `(Δfwd, Δlat, Δheight, Δyaw)` (4), qui remplace le `root_translation`
   absolu (3) — cf. `ROADMAP_DETERMINIST.md` §2.1 / §2.2.)*
3. **Z-normalization obligatoire et vérifiée**. Tout pipeline doit asserter
   post-norm ≈ N(0,1) **sur l'état, les deltas et les canaux de contrôle
   z-normalisés `vx, vz`** ; le checkpoint doit embarquer les stats du
   normalizer. Exception (ex-Q4, 2026-06-24) : les canaux de visée
   `aim_x, aim_z` sont **normés-unité par construction** — exclus de la
   z-norm et de cette assertion (pas de `mean/std`).
4. **Conditionnement multiplicatif non-bypassable** : FiLM global + per-block
   AdaLN + null embedding learnable restent **activés par défaut**.
   *(Réinterprété déterministe : le conditionnement est le signal de
   contrôle + l'embedding texte, plus le bruit cond/uncond de la diffusion.)*
5. *(historique, diffusion)* **CFG dropout obligatoire** (`condMaskProb`
   0.10–0.20) dès qu'on sample avec `cfgScale > 1`.
6. **Encodeur texte canonique : custom BPE 8k + transformer 4 layers 256d
   (~5M), output 384** (décision Pazimor 2026-06-11). Le frozen CLIP
   ViT-B/32 est conservé comme **outil de diagnostic/baseline**.
   *(Post-pivot : cet encodeur devient le composant qui donne au contrôleur
   « tout-en-un » sa capacité à recevoir des prompts — cf.
   ROADMAP_DETERMINIST.)*
7. **Sanity check overfit-1-sample** : reste le smoke test de référence
   avant tout run long.
8. **Conventions de code** : voir `doc/GUIDELINES.md` (NumPy docstrings,
   Pylance strict, pas de magic numbers, méthodes ≤ 25 lignes, fichiers
   ~500 lignes, dataclasses dans `core/types`, tests isolés).
9. **Encodeur texte découplé du moteur de génération** (décision Pazimor
   2026-06-12). L'encodeur est un **module standalone** : interface stable
   (`TextEncoderProtocol`), artefact propre (`{config.yaml, tokenizer/,
   weights.pt}`), CLI d'entraînement dédié. Le moteur le **consomme comme une
   dépendance versionnée** (chemin d'artefact + flag `encoderTrainable`).
10. **Exportabilité ONNX préservée — ne jamais bloquer la voie NPU**
    (décision Pazimor 2026-06-12). Règles : pas de control flow dépendant des
    données dans les `forward()`, pas de `.item()` / listes Python sur le
    chemin du graphe, axes dynamiques déclarés. Un test d'export tourne en
    CI ; toute PR qui le casse est refusée. *(Acquis renforcé en
    déterministe : un forward par frame, aucune boucle interne.)*

**Tension connue** : la philosophie « peu de losses » doit rester lisible.
Le refactor rend chaque composant de loss **visible, activable par config,
et observable** (poids effectif dans la loss totale).

---

## 3. Architecture globale

Le système se lit en **trois étages** : le **package Python** (recherche /
entraînement), le **modèle tout-en-un** qu'il produit, et les **plugins
moteur** qui le déploient. Cette section est la **référence d'architecture
unique** du projet ; les fiches d'exécution (`ROADMAP_DETERMINIST.md`,
`ROADMAP_PLUGINS.md`) la citent sans la redéfinir.

### 3.1 Structure du repo (recap unique)

**Seul endroit du projet qui récapitule l'arborescence complète.** Les
fiches d'exécution n'en redéfinissent aucune partie ; elles renvoient ici.
Le package Python est organisé **par couches de dépendances** (chaque module
n'importe que ce qui est strictement en dessous).

```
ai-nimator/
├── src/
│   ├── ainimator/            # package installable unique (imports: ainimator.*)
│   │   ├── core/             # L0 — types (dataclasses), constantes, schéma
│   │   │                     #   config Pydantic, device, logging. N'importe RIEN.
│   │   ├── geometry/         # L1 — quaternion, rot6d, FK, squelette SMPL-22.
│   │   ├── data/             # L2 — AMASS .npz, link dataset, preprocess,
│   │   │                     #   mirror, normalizer, séquences autorégressives.
│   │   ├── text/             # L3 — encodeur standalone (§2 #9) : tokenizer BPE,
│   │   │                     #   transformer, TextEncoderProtocol, wrapper CLIP.
│   │   ├── diffusion/        # L3 — (historique) schedule/sampler. Neutralisé.
│   │   ├── model/            # L3 — contrôleur déterministe, FiLM/AdaLN, losses
│   │   │                     #   (FK via geometry). ⛔ N'importe JAMAIS data.
│   │   ├── health/           # L4 — Probe/Contract/HealthHub, éval, audit.
│   │   ├── training/         # L4 — boucles d'entraînement. SEUL à voir data ET model.
│   │   ├── export/           # L4 — ONNX, bundle plugin, Collada/JSON, postprocess.
│   │   └── cli/              # L5 — entrypoints SANS logique : config → pipeline.
│   └── configs/             # YAML versionnés (hors package) : network, dataset,
│                            #   health, text_encoder…
├── apps/                 # déploiement moteur (Goal B, hors package Python)
│   ├── monitor/             # front-end visuel Streamlit du health JSONL (ROADMAP_MONITOR.md)
│   │                        #   livré (P0–P3) ; voie contrôleur branchée le 2026-07-01
│   ├── spec/                # contrat d'inférence partagé (source de vérité)
│   │   ├── inference_contract.md       # I/O ONNX, boucle moteur, normalisation
│   │   ├── manifest.schema.json        # schéma du manifest
│   │   ├── control_preset.schema.json  # schéma des presets de contrôle
│   │   └── reference_bundle/           # petit bundle pour tests de parité
│   ├── build/               # orchestrateur de build (vérité PLUGINS §2.9)
│   │   ├── build_plugin.py  # export bundle (--checkpoint REQUIS) → copie → build
│   │   └── Makefile         # cibles plugin-unity / plugin-unreal
│   ├── unity-sentis/        # package UPM (C#) — runtime Sentis
│   │   ├── package.json
│   │   ├── Runtime/         # ControllerRuntime / StateBuffer / Normalizer /
│   │   │                    #   ControlPreset (+ .asmdef)
│   │   ├── Editor/          # UX d'authoring (Inspector de binding)
│   │   └── Samples~/        # scène de démo
│   └── unreal-nne/          # plugin (C++/Blueprint) — runtime NNE
│       ├── AInimator.uplugin
│       ├── Source/AInimator/        # ControllerRuntime / Normalizer / ControlPreset
│       ├── Source/AInimatorEditor/  # UX d'authoring (Details / AnimGraph)
│       └── Content/                 # assets + bundle livré (gitignored)
├── doc/                     # ROADMAP*.md, GUIDELINES.md, experiments/LOG.md
├── test/                    # miroir de src/ainimator/
├── scripts/                 # orchestrateurs d'expériences

└── output/                  # runs et fichiers générés
```

Le **bundle plugin** livré (frais à chaque build, jamais committé) =
`controller.onnx + manifest.json + norm_stats.json + presets/`.

Règles d'imports (les 4 qui comptent), **vérifiées par `import-linter` en
CI** :

1. **Imports uniquement vers le bas** — jamais ascendants ; latéraux entre
   L3 seulement `model → text/diffusion`.
2. **`model` n'importe jamais `data`** : il travaille sur des tenseurs ; le
   contrat de batch est un type de `core`. (Testabilité + ONNX.)
3. **`training` est l'unique point de jonction** data ↔ model.
4. **`cli` est sans logique** : parse la config, appelle un pipeline.

Principes : **configs YAML = feature flags** (schéma Pydantic strict,
`extra="forbid"`, défauts = comportement actuel) ; **chaque run sauvegarde
sa config résolue** (`resolved_config.yaml` + git SHA + date) dans son
`outputDir` ; **mode `--debug`** sur tous les CLI d'entraînement (modèle
réduit, < 2 min) ; **smoke test overfit-1** en une commande.

### 3.2 Étage 2 — le modèle « tout-en-un »

Le moteur est un **contrôleur autorégressif déterministe** augmenté d'un
**encodeur texte intégré**. Il accepte trois entrées (toutes optionnelles et
combinables) qui passent par le **même bloc de conditionnement** FiLM/AdaLN,
et sort un `Δstate` par frame :

| Partie | Rôle | Couche |
|---|---|---|
| **encodeur texte** (custom BPE, `TextEncoderProtocol`) | prompt → embedding ; conditionnement haut-niveau (*quel* mouvement) | `text/` |
| **signal de contrôle** (`vx, vz, aim_x, aim_z` — 4 ch ; layout `ROADMAP_DETERMINIST.md §2.2`) | pilotage bas-niveau temps réel | entrée `model/` |
| **phase** (cos/sin, dérivée des contacts) | cadence locomotrice (anti-sliding) | entrée `model/` |
| **fenêtre d'état** (136 = rot6d 132 + root motion local 4 ; foot-contact/vélocités = loss-only) | contexte autorégressif | entrée `model/` |
| **cœur** (transformer autorégressif, N blocs) | régresse `Δstate` | `model/controller_v2` |
| **supervision FK** (joint_xyz, vélocités, foot-contact) | losses, hors graphe | `geometry/` + losses |

```
  prompt texte ─► text encoder ──┐
  contrôle ──────► normalizer ────┤  embeddings
  phase ──────────────────────────┤   ┌──────────────┐   ┌──────────────┐
  fenêtre d'état ─► state embed ───┴──►│ FiLM / AdaLN │──►│ contrôleur   │
                                       │ (cond.)      │   │ (transformer)│
                                       └──────────────┘   └──────┬───────┘
                                                                 ▼
   denormalize ◄─ Δstate (norm.) ◄──────────── output head ◄─────┘
        │
        ▼  intègre Δstate → état t+1 ─┐
        └──── boucle autorégressive ◄─┘  (moteur en temps réel / rollout
                                          d'entraînement — JAMAIS dans le
                                          graphe ONNX : un forward = un pas)
```

Détail — **contrat d'I/O canonique (layout état + contrôle + sortie
`Δstate`) : `ROADMAP_DETERMINIST.md` §2.2** (source unique) ; régime
d'entraînement de l'encodeur et contrats santé déterministes : §2.4, §3, §4.

### 3.3 Étage 3 — le déploiement : plugins moteur

Le modèle est exporté en **bundle ONNX auto-suffisant** et consommé tel quel
par deux plugins jumeaux (Unity Sentis, Unreal NNE) sur un **contrat
d'inférence unique**. Un seul forward par frame ; l'état et la boucle vivent
côté moteur ; foot-lock IK + blending en post. Arborescence `apps/`
(spec / build / unity-sentis / unreal-nne) et bundle livré : voir §3.1.

Le plugin pousse ses entrées vers le modèle via un **point d'entrée unique
`Entry`** à deux canaux **complémentaires et additifs** (`control` ET/OU
`prompt`) : un canal **contrôle** = vecteur `(vx, vz, aim_x, aim_z)` (le *où*,
temps réel ; `ControlPreset` nommé comme `forward`/`strafe_left` = simple
confort facultatif) et un canal **prompt** = texte → encodeur intégré (le
*quoi* expressif : danse, sit…). Ils entrent ensemble dans le conditionnement
(« danse » + `vz+1` = danse qui avance). Détail (contrat, runtime par moteur,
build, UX, phases B) : `ROADMAP_PLUGINS.md` §1.1 et §3.

---

## 4. La surcouche santé : Probe / Contract / HealthHub

`health/` est **l'outil de debug global et unique** du projet. La surcouche
s'attache au graphe `nn.Module` **via les hooks PyTorch** ; elle ne le
redéfinit pas.

| Objet | Rôle | Contrainte clé |
|---|---|---|
| `Probe` | Hooks forward/backward ; capture des **stats scalaires** (mean/std/norm/rank...) | Ne stocke **jamais** de tenseurs ; coût borné |
| `Contract` | Critères déclaratifs **en YAML** par chemin de module ; verdicts `OK / WARNING / CRITICAL` | Vocabulaire aligné sur `pytorch-auditor` |
| `HealthHub` | Registre central : attache, collecte, évalue, route vers TensorBoard + JSONL + alertes | Un seul hub, plusieurs runtimes |

Runtimes : `hub.step()` (pendant le training, 1 step sur N, défaut 50),
`hub.audit()` (checkpoint, adapteur `pytorch-auditor` niveaux 3–4),
`hub.diagnose()` (générations / rollouts), `hub.report()` (fiche santé
agrégée). CLI unique : `ainimator.cli.health {watch|audit|diagnose|report}`.

**Garde-fou anti-usine-à-gaz** : démarrer avec un jeu minimal de probes et
contracts ; tout ajout exige la justification « quel incident passé ça
aurait détecté ? ».

### 4.1 Référence des scores : sens, cible, seuils

Table canonique, encodée dans `src/configs/health.yaml` (un commentaire par
métrique). Les seuils sont calibrés sur les incidents passés ; les
recalibrer est permis mais se documente ici. *(Les lignes `cfg_sim` /
`seed_sim` sont propres à la diffusion — conservées pour référence
historique ; le déterministe utilise `control_sensitivity` / `mean_collapse`
/ `rollout_drift`, cf. ROADMAP_DETERMINIST §4.)*

| Métrique | Mesure | Direction | Sain | WARNING | CRITICAL |
|---|---|---|---|---|---|
| `fidelity` | FK-cosine(gen, GT) sur le probe set | ↑ vers 1 | > 0.60 | 0.20–0.60 | < 0.20 |
| `retrieval` | fraction où nearest-GT(gen_i) == GT_i | ↑ vers 1 | ≥ 0.80 | 0.40–0.80 | ≤ chance |
| `distinctness` | FK-cosine moyen inter-générations | ↓ vers 0 | < 0.50 | 0.50–0.80 | > 0.80 |
| `conditioning_sensitivity` | Δloss forward conditionnement réel vs shuffled | ↑ > 0 | > 0 et stable | ≈ 0 sur 1 fenêtre | ≈ 0 persistant |
| `effective_rank` | rank effectif des hidden states (par bloc) | ↑ | > 0.5 × D | 0.1–0.5 × D | < 0.1 × D |
| `intra_batch_sim` | cosine moyen des hidden states intra-batch | ↓ | < 0.5 | 0.5–0.9 | > 0.9 |
| `update_ratio` | ‖Δw‖/‖w‖ par couche et par step | plage | 1e-4 – 1e-2 | hors plage | ≈ 0 persistant ou NaN |
| `loss_share` | part pondérée de chaque composant | équilibre | 1–70 % | < 1 % ou > 80 % | exactement 0.0 |
| `post_norm_stats` | mean/std par channel après z-norm (état, deltas, contrôle `vx,vz` ; **hors** `aim_x,aim_z` normés-unité) | mean→0, std→1 | \|mean\|<0.1, \|std−1\|<0.1 | < 0.3 | au-delà |
| `nan_inf` | scan NaN/Inf poids + activations + gradients | 0 | 0 | — | ≥ 1 |
| `val_gap` | (val − train)/train | ↓ | < 15 % | 15–40 % | > 40 % |
| `epoch_time` / `rss_memory` | temps/epoch et mémoire résidente | stable | dérive < 1.5× | 1.5–3× | > 3× |

Notes d'interprétation : **fidelity ≈ 0 = aléatoire** (le succès se joue
entre 0.6 et 1) ; **retrieval se lit contre la chance** (1/n_probes) ;
**distinctness bas n'est bon que si fidelity est haut** — lire les trois
ensemble (rôle de `hub.report()`).

---

## 5. Répartition des rôles

> **Roster (2026-06-25)** : **cinq agents** dans `.claude/agents/` —
> `orchestrator`, `dev-python-neural`, `reviewer`, `dev-unity-plugin`,
> `dev-unreal-plugin` — plus **Pazimor** (humain). L'ancien rôle
> `experimenter` a été **supprimé** : son analyse de runs/régressions est
> reprise par l'**orchestrator** (Opus) + Pazimor ; le workflow
> `doc/experiments/requests/` subsiste, servi par `dev-python-neural`.

| Acteur | Périmètre | Interdits |
|---|---|---|
| **orchestrator** (Opus) | Vision globale : décompose un prompt, route vers le bon spécialiste, **analyse runs/régressions** (mean collapse, drift, sensibilité au conditionnement), dépose des demandes dans `doc/experiments/requests/`. Lit tout. | Écrire du code ; trancher les vérités §2 / `TALK_QUESTIONS` |
| **dev-python-neural** (Sonnet, Python / réseaux) | Création et entraînement du réseau de neurones : contrôleur déterministe, encodeur texte intégré, losses, health/, export ONNX, tests. Outillage Python demandé par les autres rôles ; sert les demandes `doc/experiments/requests/`. | Changer les défauts d'hyperparamètres ; toucher aux vérités §2 ; lancer des runs > 30 min sans demande ; écrire du code moteur (C#/C++) |
| **reviewer** (lecture seule) | Vérifie critères d'acceptation d'une phase + conventions (`doc/GUIDELINES.md`) avant de la déclarer terminée | Modifier des fichiers |
| **dev-unity-plugin** (C#/Sentis) | `apps/unity-sentis/` uniquement (ROADMAP_PLUGINS) | Hors de son sous-projet ; ML/réseau |
| **dev-unreal-plugin** (C++/NNE) | `apps/unreal-nne/` uniquement (ROADMAP_PLUGINS) | Hors de son sous-projet ; ML/réseau |
| **Pazimor** | Arbitrages, achat compute, validation visuelle Blender, mise à jour des fiches | — |

**Maintenance** : toute expérience conclue ajoute une ligne dans
`doc/experiments/LOG.md` (date, run, config, métriques, verdict) ; toute
décision nouvelle modifie la section concernée ici, avec date.

---

## 6. Références

- `doc/ROADMAP_DETERMINIST.md` — **Goal A**, moteur déterministe (contrôleur
  + encodeur texte intégré, phases A0→A7, schéma du modèle).
- `doc/ROADMAP_PLUGINS.md` — **Goal B**, plugins Unity / Unreal (phases B0→B6).
- `doc/TALK_QUESTIONS.md` — questions ouvertes à trancher en discussion
  (création du dataset, face tracking). *(Les anciennes entrées — contrat
  d'I/O, profils, stats de contrôle, export — sont tranchées et reportées
  dans les fiches.)*
- `doc/ROADMAP_MONITOR.md` — moniteur visuel de santé (`apps/monitor/`,
  front-end Streamlit du health JSONL, découplé).
- `doc/GUIDELINES.md` — conventions de code (paragraphes à ID stable).
- `doc/experiments/LOG.md` — journal des runs (une ligne par expérience).
- Outils : skill `pytorch-auditor` (audit checkpoint), skill `ai-debugging`
  (méthodologie), `ainimator.cli.health` (debug global).
