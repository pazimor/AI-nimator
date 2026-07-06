# AI-nimator — Fiche de route « Goal A : contrôleur déterministe »

> **Statut** : document de référence (canonical) pour le **Goal A**, le
> **moteur de génération unique** du projet — contrôleur déterministe et
> autorégressif (full déterministe depuis le pivot 2026-06-23 ; la diffusion
> est abandonnée). Il ne remplace pas `doc/ROADMAP.md` : il
> le **complète**. Toute décision qui contredit ce document doit être tranchée
> par l'utilisateur, puis reportée ici.
>
> **Note historique** : ce document a été créé (2026-06-22) pour une voie
> déterministe *parallèle* à la diffusion ; le pivot du 2026-06-23 en a fait
> la **seule** voie. Les mentions « parallèle / deuxième moteur » résiduelles
> sont historiques.
>
> **Lire d'abord** : `doc/ROADMAP.md` (fiche de cadrage du projet : vérités,
> architecture, santé, rôles — full déterministe depuis 2026-06-23). Ce
> document réutilise sans les redéfinir : l'architecture en couches (§3 de
> ROADMAP), la surcouche santé Probe/Contract/HealthHub (§4), la table des
> scores (§4.1), le contrat d'imports `import-linter`, les conventions de
> code (`doc/GUIDELINES.md`, vérité §2 #8) et la
> répartition des rôles (§6).
>
> **Public** : agent d'implémentation (Sonnet) pour le code, agent
> d'expérimentation (Opus) pour le débogage des régressions, l'utilisateur pour
> les arbitrages.
>
---

## 1. le GOAL en une page
Le **Goal A** vise : un **contrôleur autorégressif**
qui produit l'animation **une frame à la fois**, piloté par un **signal de
contrôle** (vitesse désirée, direction de visée, text),
pour un usage **temps réel** dans un moteur de jeu (Unity Sentis / Unreal
NNE).

Idée centrale : on *régresse* le mouvement. `f(state_t, control_t, [prompt_emb], [phase]) → Δstate`. Un seul forward par frame, aucune boucle de débruitage — ce qui rend l'export ONNX/NPU trivial. L'**embedding texte est calculé en amont** de la boucle (un passage encodeur par prompt, **pas** par frame), puis injecté comme entrée du graphe à chaque forward (cf. §2.4) ; il reste donc hors du chemin temps réel par frame.

**Principes** (identique à ROADMAP) : tout reste dans le
package `ainimator`. Le contrôleur est une **variante de la couche
`model/`**, piloté par `src/configs/network.yaml`, gardé par le smoke test
overfit-1, instrumenté par `HealthHub`. squelette d'abord, on valide chaque étape à
l'overfit avant d'avancer.

**Décisions de cadrage actées**:

- **Modèle « tout-en-un » avec encodeur texte intégré (2026-06-23)** : le
  contrôleur reçoit, en plus du signal de contrôle, un **prompt texte**
  encodé par l'encodeur custom BPE (réutilisé du pipeline existant, §2.6 de
  ROADMAP — il n'est PAS perdu). Le prompt fournit un conditionnement de
  haut niveau (« quel mouvement »), le signal de contrôle pilote le
  bas-niveau temps réel (« à quelle vitesse / dans quelle direction »). Les
  deux entrées sont optionnelles et combinables (cf. §2.4 et schéma §3.0).
- **Phase explicite d'abord** : on démarre avec une phase locomotrice
  *fournie en entrée* (signal explicite), puis on ouvre la porte à une
  phase apprise via un flag (`phase: none|explicit|learned`). La phase est
  traitée comme **non optionnelle** — c'est le seul point où l'on peut
  encore se planter sérieusement (cf. §7).
- **Style (`z_style`) différé** (l'utilisateur, 2026-06-22) : le dataset actuel
  (AMASS + captions HumanML3D/KIT) **n'a pas de labels de style**
  exploitables (pas de sous-set ninja/businessman). L'injection de style
  est donc **hors du chemin critique** : flag `style-latent: false` par
  défaut, phase A3 marquée *optionnelle/différée*, conditionnée à
  l'obtention d'un dataset labellisé style. Le conditionnement du
  contrôleur repose, à ce stade, sur les **seuls vecteurs de contrôle**
  (vitesse désirée + direction de visée).

---

## 2. Vérités canoniques du déterministe

Décisions actées propres au Goal A. Elles **s'ajoutent** aux vérités §2 de
ROADMAP ; elles n'en abrogent aucune. Sonnet ne doit PAS les re-débattre ;
si un blocage technique l'exige, escalader à l'utilisateur.

### 2.1 Ce qui est partagé, hérité, nouveau, ou neutralisé localement

| Vérité ROADMAP §2 | Sort dans Goal A |
|---|---|
| #1 v-pred + cosine + DDIM + Min-SNR | **Sans objet** : pas de schedule ni de sampling dans le contrôleur. Diffusion abandonnée au pivot (2026-06-23) — acquis historique inactif. |
| #2 Lean representation 135 ch (rot6d 132 + root_trans 3) | **Hérité et étendu** : l'état **régressé** du contrôleur fait **136** = rot6d (132) + **root motion local** `(Δfwd, Δlat, Δheight, Δyaw)` (4), qui remplace le `root_translation` absolu (3). Les signaux FK-dérivables (foot-contact, vélocités) restent supervisés à la loss. Layout figé : §2.2. |
| #3 Z-normalization obligatoire et assertée | **Hérité et étendu (ex-Q4, 2026-06-24)** : post-norm ≈ N(0,1) asserté sur l'état, les deltas **et les canaux de contrôle `vx, vz`**. `aim_x, aim_z` = **normés-unité par construction**, exclus de la z-norm (pas de `mean/std`). Stats du normalizer embarquées dans le checkpoint. |
| #4 Conditionnement non-bypassable (FiLM/AdaLN/null) | **Réinterprété** : le conditionnement n'est plus le texte mais le *signal de contrôle* + `z_style`. Mêmes leviers (FiLM/AdaLN), même exigence de non-bypass (cf. contrat `control_sensitivity`, §4). |
| #5 CFG dropout si cfgScale > 1 | **Sans objet** : pas de CFG en déterministe. |
| #6 Encodeur texte custom BPE canonique | **Hérité et intégré** (2026-06-23) : l'encodeur custom BPE est désormais un **composant du modèle tout-en-un** (cf. §2.4). Il fournit le conditionnement texte du contrôleur, consommé via `TextEncoderProtocol` (§2.9 ROADMAP) — module découplé, jamais redéfini par le contrôleur. |
| #7 Smoke test overfit-1-sample | **Hérité, ré-instrumenté** : overfit 1 séquence + rollout qui se reproduit (cf. A1). |
| #8 Conventions de code (§2.8) | **Hérité tel quel** : NumPy docstrings, Pylance strict, pas de magic numbers, méthodes ≤ 25 lignes, fichiers ~500 lignes, dataclasses dans `core/types`, tests isolés. |
| #9 Encodeur texte découplé | **Respecté** : si texte→style, dépendance versionnée, jamais d'import remontant. |
| #10 Exportabilité ONNX (voie NPU) | **Promu d'acquis** : un contrôleur = un seul forward par frame, aucune boucle interne. Les règles §2.10 (pas de control flow data-dépendant, pas de `.item()`, axes dynamiques) restent la loi ; le test d'export CI s'étend au contrôleur (cf. A5). |

### 2.2 Contrat d'I/O déterministe (source unique) — état + contrôle

> **Source unique du contrat d'I/O du modèle.** Le layout exact des entrées
> (vecteur d'état, signal de contrôle, phase) et de la sortie (`Δstate`) est
> défini **ici et seulement ici**. Les autres fiches ne le redéfinissent pas :
> `ROADMAP.md §3.2` (schéma des parties) et `ROADMAP_PLUGINS.md §3.1`
> (`manifest.json`) **citent cette section**. Le `manifest.json` du bundle
> (Goal B) est la **sérialisation** de ce contrat, pas une seconde définition.

#### 2.2.a Vecteur d'état (l'invariant à régresser)

C'est l'équivalent du *schema* lean de la diffusion, et son **invariant**.
À figer en A0, avant tout code, dans `core/types` + `core/constants`.

- **Socle hérité (lean v2)** : `rotation6d` (132 ch).
- **Root motion local (régressé)** : le déplacement du root est régressé en
  **repère local du personnage**, par frame, **PAS** en position monde
  absolue (qui dérive à l'infini et casse la z-norm / l'autorégression) :
  `Δforward`, `Δlateral`, `Δheight` (3) + `Δyaw` (1) = **4 canaux**. La
  transform monde est intégrée **côté moteur, hors graphe ONNX** (§2.2.c).
  Ceci remplace le `root_translation` absolu (3) du socle lean.
- **Décision (2026-06-24)** : l'état régressé est donc
  **136 canaux** = `rotation6d` (132) + root motion local (4). **Rien
  d'autre** n'est empilé en sortie.
- **Foot-contact : loss-only.** Les labels de contact pied sont
  **supervisés à la loss** (FK via `geometry/`), **jamais** sortis comme
  canaux prédits. Le foot-lock IK runtime **redérive** les contacts de la
  pose côté moteur. (Tranché 2026-06-24 ; garde la sortie lean.)
- **Vélocités articulaires** : supervisées à la loss uniquement (FK-dérivables).
- **Phase locomotrice** : *explicite* au départ (signal fourni en **entrée**,
  cf. bus de conditionnement §2.2.d), flag pour basculer en *apprise*.
- **Style** : voie principale = **adaptateur LoRA** (extensibilité au niveau
  poids/export, cf. §2.2.e), **pas** un canal d'état. La piste `z_style`
  (conditionnement) reste une **option différée** réservée au seul cas
  « dataset labellisé style + besoin de blending de style en direct »
  (phase A3 optionnelle, §5) ; non régressée à ce stade.

Règle d'or (cohérente avec §2 de ROADMAP) : tout signal **FK-dérivable**
est supervisé **à la loss** via FK (`geometry/`), jamais empilé comme
channel prédit redondant. Le contrôleur prédit `Δstate` ; les contacts et
vélocités servent de cibles de supervision et/ou de conditionnement, pas
de sorties dupliquées.

#### 2.2.b Signal de contrôle (l'entrée de pilotage temps réel)

Le signal de contrôle pilote le bas-niveau (« à quelle vitesse / dans quelle
direction »), par opposition au prompt texte qui fixe le haut-niveau (§2.4).

**Layout figé (2026-06-24) — 4 canaux, liste ordonnée :**

```
control = (vx, vz, aim_x, aim_z)
          ├── (vx, vz)        vitesse désirée planaire, repère local sol (X-Z, Y-up)
          └── (aim_x, aim_z)  direction de visée 2D normalisée
```

Le `aim` est **vectoriel 2D** (et non scalaire/cap) : choisi pour rester
**découplé** de l'orientation de locomotion — viser dans une direction tout
en se déplaçant dans une autre (prérequis du futur groupe `interaction`,
armes). Pour l'entraînement, ce contrôle est **dérivé de la GT** (cf. A1/A6).

#### 2.2.c Sortie

Le contrôleur produit un **`Δstate` normalisé** par frame (un seul forward).
L'intégration `Δstate → état t+1` vit hors du graphe ONNX (boucle moteur /
rollout d'entraînement, §3.0 ROADMAP).

> ✅ **TRANCHÉ (2026-06-24).** Les deux points
> autrefois ouverts sont figés :
>
> 1. **Signal de contrôle = 4 canaux** `(vx, vz, aim_x, aim_z)` : vitesse
>    désirée planaire locale (2) + direction de visée 2D normalisée (2). Le
>    `aim` est **2D** (pas scalaire/cap), pour rester découplé de
>    l'orientation de locomotion. → `core/constants` peut figer ces indices.
> 2. **État régressé = 136 canaux** : `rotation6d` (132) + root motion local
>    `(Δforward, Δlateral, Δheight, Δyaw)` (4). Foot-contact et vélocités
>    articulaires = **loss-only** (FK-dérivables, redérivés côté moteur), pas
>    de canaux de sortie. `manifest.json` (B0) peut sérialiser ce layout.
>
> Les exemples des autres fiches doivent s'aligner sur ce layout : PLUGINS
> §1.1 (`(vx, vy)` / `aim=0` scalaire) et §3.1 sont **mis à jour** pour citer
> `(vx, vz, aim_x, aim_z)`.

#### 2.2.d Bus de conditionnement extensible — entrées optionnelles, sortie figée

> **Principe anti-blocage (l'utilisateur, 2026-06-24).** La **sortie `Δstate` (136)
> est l'invariant figé** : c'est elle qu'épinglent l'export ONNX et le
> `manifest.json`. Toute capacité nouvelle (objets, décor, impacts, gabarit)
> entre par les **entrées de conditionnement**, jamais par un canal de
> sortie. Ajouter une feature = élargir le bus d'entrée + bumper la version
> *d'entrée* du manifest ; la signature de sortie ONNX ne bouge **jamais**.

Toutes ces entrées passent par les mêmes leviers FiLM/AdaLN que le signal de
contrôle (§2.4), sont **optionnelles** (null embedding learnable si absentes)
et **calculées en amont** du `forward()` (aucune boucle ni control flow
data-dépendant — règle ONNX §2.10). Quatre groupes sont **réservés dès
maintenant** (rubrique nommée dans le manifest, vide tant que non câblée) ;
seuls `control`/`phase`/`prompt` sont actifs en A1–A6.

| Groupe | Famille | Rôle | Statut |
|---|---|---|---|
| `control` | pilotage | vitesse désirée + visée (`vx,vz,aim_x,aim_z`) | actif (A1) |
| `phase` | pilotage | cadence locomotrice (anti-sliding) | actif (A2) |
| `prompt` | intention | conditionnement texte haut-niveau (§2.4) | actif (modèle tout-en-un) |
| `interaction` | monde→corps | objets manipulables : ramasser, taper avec une arme (pose/orientation relative, type, état type « gâchette ») | **réservé** |
| `perception` | monde→corps | décor : « voir » et se repérer (heightfield terrain, obstacles locaux) — adaptation pied sur marche, passage de biais | **réservé** |
| `reaction` | monde→corps | forces/impacts externes reçus : se faire toucher, pousser, knockback, rééquilibrage (dual de `interaction`) | **réservé** |
| `morphology` | soi | dimensions du corps (taille, membres, gabarit / `betas` SMPL) — retargeting multi-tailles, jugement de gabarit | **réservé** |

Garde-fou anti-usine-à-gaz : on s'arrête à ces 4 groupes réservés. Les
besoins voisins se replient sur l'existant (goal/look-at → `control`,
émotion/fatigue → LoRA ou `prompt`) ; tout nouveau groupe exige la
justification « quel besoin concret il débloque ». Réserver une rubrique
**vide** coûte zéro ; câbler un contrat ne se fait qu'au besoin réel.

> ⚠ Aucun de ces groupes n'est livré en A1–A7 : ils ne sont câblés que
> lorsqu'un dataset les portant existe. Ils figent seulement le **contrat
> d'extensibilité** pour que les ajouts futurs ne cassent ni l'export ni le
> manifest.

#### 2.2.e Style & personnalisation par adaptateur LoRA (extensibilité poids — V10+)

> **Décision de cadrage (l'utilisateur, 2026-06-24).** Le style et la
> personnalisation « bring your own animations » passent par un **adaptateur
> LoRA** (adaptation de poids), **pas** par un canal d'I/O. C'est orthogonal
> au contrat de §2.2 : LoRA ne touche **aucun canal**.

Mécanisme : poids de base gelés + mises à jour bas-rang `B·A` sur les
`nn.Linear` standards du contrôleur (attention q/k/v/o, MLP — déjà présents).
L'utilisateur entraîne un petit adaptateur sur son propre set d'animations
(aucun label de style requis dans le dataset de base).

**Compatibilité ONNX — non bloquante, voie claire** : l'entraînement LoRA vit
côté PyTorch (autograd). Pour l'export, on **fusionne** l'adaptateur dans les
poids (`W' = W + B·A`) **avant** l'export ONNX → le **graphe est identique**,
seules les valeurs de poids changent (coût runtime nul, zéro complication
ONNX). Un style = un `.onnx` mergé (ou base + adaptateur mergés au build). La
voie « adaptateur séparé hot-swappable dans le graphe » est écartée
(initializers ONNX figés, friction inutile).

Coût aujourd'hui pour ne pas se bloquer : **nul** — garder les couches
adaptables en `nn.Linear` (déjà le cas) et prévoir une **étape optionnelle
« merge adaptateur → export »** dans le pipeline d'export (bundle Goal B,
cf. `ROADMAP_PLUGINS.md`). Rien à réserver dans le contrat I/O.

Complémentarité avec `z_style` (§2.2.d) : LoRA = personnalisation hors-ligne,
fichier adaptateur minuscule et partageable, peut aussi porter de **nouveaux
comportements** ; `z_style` resterait pertinent seulement pour du **blending
de style continu en direct** (curseur par frame) *si* un dataset labellisé
apparaît. Sur le petit modèle actuel (256d/4L), le gain compute de LoRA est
marginal — l'intérêt est le **petit artefact partageable**, pas le coût
d'entraînement.

### 2.3 Le contrôleur est une variante de `model/`, piloté par config

Le moteur de génération est sélectionné par le champ `model-type` dans
`src/configs/network.yaml` (cf. §3.2). **Cohérence de nommage à garder** :
toutes les clés YAML sont en **kebab-case** (`model-type`, jamais
`model.type` ni `model_type`) ; c'est le nom canonique, utilisé tel quel par
le loader et le CLI. Diffusion et contrôleur coexistent ;
`training/` reste l'unique point de jonction `data ↔ model` ; `cli/` reste
sans logique. Aucun nouveau package, aucune entorse au contrat d'imports.

### 2.4 Encodeur texte intégré — le modèle « tout-en-un » (2026-06-23)

> **Source unique des deux méthodes de guidance.** La définition **niveau
> modèle** des deux entrées de pilotage — **prompt texte** (haut-niveau) et
> **vecteur de contrôle** (bas-niveau temps réel), additives et optionnelles —
> vit **ici et seulement ici** (layout du contrôle : §2.2.b ; bus de
> conditionnement : §2.2.d). Les autres fiches ne la redéfinissent pas :
> `ROADMAP_PLUGINS.md §1.1/§2` **cite cette section** et ne décrit que
> l'**exposition plugin** (assets `ControlPreset`, point d'entrée `Entry`,
> bindings éditeur).

Le contrôleur n'est plus piloté par le seul vecteur de contrôle : il
accepte aussi un **prompt texte**. L'encodeur custom BPE (couche `text/`,
§2.6 ROADMAP, **réutilisé tel quel** — il n'est pas perdu) transforme le
prompt en un **embedding de conditionnement** injecté dans le contrôleur par
les mêmes leviers FiLM/AdaLN que le signal de contrôle.

Rôles complémentaires des deux entrées :

| Entrée | Encodage | Rôle | Optionnel ? |
|---|---|---|---|
| **Prompt texte** | encodeur BPE (`TextEncoderProtocol`) → embedding | conditionnement **haut-niveau** : *quel* mouvement (« marche fatiguée », « saut ») | oui (null embedding learnable si absent) |
| **Signal de contrôle** | `vx,vz` z-normalisés / `aim_x,aim_z` normés-unité → vecteur (ex-Q4) | pilotage **bas-niveau temps réel** : vitesse désirée, direction de visée | oui (zéro = laisser le prompt décider) |
| **Phase** | dérivée des contacts (PFNN-style) | cadence locomotrice (anti-sliding) | flag `phase` |

Règles (héritées de §2.9 / §2.10 ROADMAP) : l'encodeur reste un **module
découplé** (artefact versionné, swap custom↔CLIP par config, jamais
d'import remontant) ; son intégration n'introduit **aucune boucle ni control
flow data-dépendant** dans le `forward()` du contrôleur — l'embedding texte
est calculé en amont (un seul passage encodeur par prompt, pas par frame),
exporté comme une **entrée du graphe** ONNX. Le régime d'entraînement de
l'encodeur (joint vs pré-entraîné puis gelé/fine-tuné) reste tranché
expérimentalement (cf. A6).

---

## 3. Architecture cible (Goal A)

> **Schéma du modèle et vue d'ensemble des parties : `ROADMAP.md` §3.2**
> (architecture globale, étage 2). Cette fiche ne décrit ici que les
> spécificités d'exécution du contrôleur (placement fichiers, flags config,
> styles différés) ; le flux entrées → conditionnement → cœur → `Δstate` →
> boucle autorégressive et la supervision FK y sont consolidés une fois.

### 3.1 Placement dans les couches (§3 de ROADMAP, inchangé)

> **Arborescence du repo : `ROADMAP.md` §3.1** (recap unique). Le
> déterministe ne crée **aucune nouvelle couche** ; il ajoute des fichiers
> dans les couches existantes, en respectant « imports vers le bas » et
> « `model` n'importe jamais `data` ».

Fichiers ajoutés / réutilisés par le contrôleur, couche par couche :

- `core/types` : `ControllerState`, `ControlSignal`, `StylePreset`
  (dataclasses) ; `core/constants` : indices et noms de canaux de l'état.
- `core` (schéma) : `ControllerV2Config` (Pydantic, `extra="forbid"`).
- `geometry` *(inchangé)* : FK réutilisée pour la supervision contacts /
  vélocités et l'intégration `root_translation`.
- `data` : builder de séquences autorégressives (fenêtres, cibles `Δstate`,
  labels contact, signal de phase explicite, contrôle dérivé de la GT).
- `text` *(inchangé)* : encodeur consommé pour le conditionnement texte
  (modèle tout-en-un, §2.4).
- `model` : ★ `controller_v2.py` (`MotionController` autorégressif,
  FiLM/AdaLN réutilisés de `layers/`) ; `losses_controller_v2.py` (L2
  vitesses + géodésique 6D + velocity + foot-contact) ;
  `style_latent.py` *(DIFFÉRÉ A3)*.
- `health` : nouveaux contrats (§4), réutilise Probe/Hub.
- `training` : boucle autorégressive (rollout, scheduled sampling rampé).
- `export` : export ONNX d'un forward contrôleur (A5).
- `cli` : `train_controller_v2`, `generate_controller_v2`,
  `eval_controller_v2`, `export_onnx controller` (sans logique).
- `configs/network.yaml` : bloc `model-type: controller` (§3.2) ;
  `configs/styles/` *(DIFFÉRÉ A3)*.

Le contrat `import-linter` est étendu, pas réécrit : `model/controller_v2`
peut importer `core`, `geometry`, `text`, `diffusion` (comme `denoiser_v2`)
mais **jamais** `data`. Un test négatif est ajouté (sur le modèle du test d'imports existant : un import remontant casse `lint-imports`).

### 3.2 `network.yaml` — le sélecteur + les profils (config-first)

> **Décision de cadrage (l'utilisateur, 2026-06-23) — config-first, zéro flag
> d'hyperparamètre.** Un run se configure par un **profil nommé** vivant en
> YAML, **pas** par des flags CLI. La surface CLI se réduit à
> `--profile {overfit|full|debug}` (+ `--config <path>` et le strict minimum
> non-config : chemins d'I/O, `--resume`). Tout ce qui était un flag
> d'hyperparamètre (`--num-clips`, `--held-out-clips`, `--min-frames`,
> `--embed-dim`, `--num-heads`, `--num-layers`, `--phase`,
> `--foot-contact-weight`, `--scheduled-sampling`, `--dataset-root`…) devient
> un **champ de profil** dans la config. Bénéfice : un run est entièrement
> décrit par son `resolved_config.yaml`, rejouable et diffable ; plus de
> divergence flag↔YAML. `debug` est un **profil**, pas un flag.

Le projet utilise `src/configs/network.yaml` (pas `features.yaml`). On y
ajoute un sélecteur de moteur, un bloc `controller`, et un bloc `profiles`
(overfit / full / debug), en kebab-case comme le reste du fichier. Défauts
choisis = comportement déterministe minimal et sûr (phase explicite,
scheduled sampling à 0). **Aucun défaut d'hyperparamètre existant n'est
modifié** (règle §2.8 / §6).

```yaml
v2:
  generation:
    # Sélecteur de moteur. "diffusion" = comportement actuel (défaut).
    # "controller" = Goal A. Le CLI de training lit ce champ.
    model-type: diffusion          # diffusion | controller

    controller:
      autoregressive: true
      phase: explicit              # none | explicit | learned
      # DIFFÉRÉ : pas de labels de style dans le dataset actuel (§1).
      # Reste false jusqu'à A3 (dataset labellisé style requis).
      style-latent: false
      # Fenêtre de contexte autorégressif (frames vues par forward).
      context-frames: 1            # étendue en A4 (robustesse longue)
      losses:
        velocity-loss: true        # flag déjà présent côté v2
        foot-contact-loss: true
        geodesic-rotation: true
      training:
        # Rampe scheduled sampling 0 → cible (corrige l'exposure bias).
        # Reste à 0 jusqu'à A4 (validation rollout court d'abord).
        scheduled-sampling: 0.0

    # Profils nommés : SEULE surface de configuration d'un run (config-first).
    # Le CLI sélectionne un profil ; il ne porte aucun hyperparamètre en flag.
    profiles:
      overfit:                      # smoke test canonique (1 clip), vérité #7
        num-clips: 1
        held-out-clips: 0
        embed-dim: 256
        num-heads: 4
        num-layers: 4
      full:                         # généralisation (split train/held-out, A6)
        num-clips: 128
        held-out-clips: 16
        min-frames: 60
        embed-dim: 256
        num-heads: 4
        num-layers: 4
        scheduled-sampling: 0.3     # warm-start A4 recommandé (--resume)
      debug:                        # bout-en-bout < 2 min sur MPS (mode debug rapide)
        num-clips: 1
        embed-dim: 64
        num-heads: 1
        num-layers: 1
        health-every-steps: 1
```

> ⚠ Les valeurs ci-dessus sont **indicatives** : la liste finale des champs
> migrés depuis les flags reste à arrêter (cf. Q2 point 2,
> `doc/TALK_QUESTIONS.md`).

> **Décisions de l'utilisateur (2026-06-24) — Q2 points 1 & 3.**
>
> 1. **Un profil est la SEULE unité de config d'un run — règle dure.** Les
>    trois profils `overfit / full / debug` suffisent pour le moment. Tout
>    nouveau besoin (test, échelon de capacité A6, variante d'archi…) se
>    matérialise **OBLIGATOIREMENT par un nouveau profil nommé** ajouté ici.
>    Interdiction d'introduire un flag d'hyperparamètre ou un override ad hoc
>    en CLI pour « juste tester » : si ça change un run, ça vit dans un profil.
> 2. **Base + override (héritage DRY).** Un profil n'est **pas autonome** : il
>    part d'une **`base`** (la fusion des configs canoniques — `network.yaml`
>    et les autres YAML du run) et n'exprime qu'un **delta**. Un profil peut
>    donc **override des champs définis dans d'autres YAML** (p. ex.
>    `dataset.yaml`), pas seulement ceux de `network.yaml`. Résolution :
>    `base (YAML canoniques fusionnés)` → `delta du profil` →
>    `resolved_config.yaml` (trace complète, G-RESOLVEDCONFIG). `--config <path>`
>    peut pointer un YAML hors `network.yaml` pour fournir/compléter cette
>    base.

Comme tout run v2, un run contrôleur écrit son `resolved_config.yaml`
(config complète + git SHA + date) dans son `outputDir` (G-RESOLVEDCONFIG).

> **Q2 point 4 — tranché (2026-06-24).** Les flags de `health diagnose`
> (`--control`, `--rollout-frames`, `--shuffle-control`, `--phase-mode`, et
> les actuels `--prompt`, `--frames`, `--seeds`…) **migrent aussi en profils**
> (section `profiles` dédiée au diagnostic). Aucune exception : la même
> mécanique base+delta s'applique. La CLI doit rester triviale —
> `health diagnose --profile <nom> <checkpoint>` — pas de réglage de
> diagnostic en flag. Voir guideline `G-PROFILES`.

#### Mapping exhaustif flags → champs (Q2 point 2, tranché 2026-06-24)

Union des flags des CLI contrôleur actuelles (`train_controller_v2`,
`…_multiclip_v2`, `…_generalization_v2`, `generate_controller_v2`,
`health diagnose`). Cible : **un seul** `train_controller_v2 --profile …`
(A7). Colonne « cible » = où le champ atterrit dans `network.yaml`.

| Flag actuel | Cible (config) |
|---|---|
| `--num-clips` | profil → `data.num-clips` |
| `--held-out-clips` | profil → `data.held-out-clips` |
| `--min-frames` | profil → `data.min-frames` |
| `--clip-batch-size` | profil → `training.clip-batch-size` |
| `--eval-sample-clips` | profil → `training.eval-sample-clips` |
| `--epochs` | profil → `training.epochs` |
| `--learning-rate` | profil → `training.learning-rate` |
| `--scheduled-sampling` | profil → `controller.training.scheduled-sampling` |
| `--foot-contact-weight` | profil → `controller.losses.foot-contact-weight` |
| `--context-frames` | profil → `controller.context-frames` |
| `--phase` | profil → `controller.phase` |
| `--aim-direction` | profil → `controller.aim-direction` |
| `--embed-dim` | profil → `controller.arch.embed-dim` |
| `--num-heads` | profil → `controller.arch.num-heads` |
| `--num-layers` | profil → `controller.arch.num-layers` |
| `--seed` | profil → `training.seed` |
| `--dataset-root` | **base** (`dataset.yaml`), override possible par profil |
| `--health-every-steps` | profil → `training.health-every-steps` |
| `--fps` (génération) | profil → `generate.fps` |
| `--repeat` (génération) | profil → `generate.repeat` |
| `--reinject` (génération) | profil → `generate.reinject` |
| `--smooth` (génération) | profil → `generate.smooth` |
| `--control` (diagnose) | profil → `diagnose.control` |
| `--rollout-frames` (diagnose) | profil → `diagnose.rollout-frames` |
| `--shuffle-control` (diagnose) | profil → `diagnose.shuffle-control` |
| `--phase-mode` (diagnose) | profil → `diagnose.phase-mode` |
| `--prompt` / `--frames` / `--seeds` (diagnose) | profil → `diagnose.*` |

**Restent en flag CLI** (non-config, par invocation — pas des
hyperparamètres) : `--output-dir`, `--checkpoint`, `--output`, `--dae` /
`--dae-gt` (chemins d'I/O), `--resume`, `--config`, `--profile`, et
`--device` (runtime/environnement). Tout le reste = profil. Détail de la
règle : guideline `G-PROFILES`.

### 3.3 Registre de styles `configs/styles/` — DIFFÉRÉ (A3)

> ⚠ **Non réalisable avec le dataset actuel.** AMASS + captions
> HumanML3D/KIT n'exposent aucun label de style. Cette sous-section décrit
> la cible *si* un dataset labellisé style devient disponible ; rien ici
> n'est livré en A1–A2, et `style-latent` reste `false`.

`z_style` serait **verrouillable** : « lock » = geler le vecteur de style.
Chaque preset = un fichier versionné (`ninja.yaml`, `businessman.yaml`, …)
référencé par nom dans la config de génération. Le contrôleur apprendrait
les vecteurs sur le sous-set labellisé style. Désentanglement exigé : le
style change le **comment**, jamais le **quoi** (un ninja avance encore
correctement). Vérifié par les contrats `style_separation` /
`style_leakage` (§4), eux aussi différés.

---

## 4. Extensions HealthHub (Goal A)

Les trois objets de la surcouche santé (Probe / Contract / HealthHub,
verdicts `OK / WARNING / CRITICAL`, adapteur `pytorch-auditor`) **ne
bougent pas** (§4 de ROADMAP). On ajoute des **contrats spécifiques
déterministe**, déclarés dans `src/configs/health.yaml`, chacun portant en
commentaire sa ligne de la table ci-dessous (direction, cible, seuils),
exactement comme la table §4.1 de ROADMAP.

Réinterprétation clé : en diffusion, le « collapse » = le prompt est
ignoré. En déterministe, l'analogue = **le contrôleur sort la même chose
quel que soit le contrôle** (insensibilité au contrôle) ou **régresse vers
la moyenne** des mouvements. Les sondes `effective_rank` et
`intra_batch_sim` (déjà existantes) sont **réutilisées telles quelles**,
seulement réinterprétées.

| Métrique (contrat) | Mesure | Direction | Sain | WARNING | CRITICAL | Détecte |
|---|---|---|---|---|---|---|
| `control_sensitivity` | Δ de sortie quand on **shuffle le signal de contrôle** (port direct de la sonde de sensibilité au conditionnement diffusion) | ↑ > 0 | nettement > 0 et stable | ≈ 0 sur 1 fenêtre | ≈ 0 persistant (contrôle ignoré) | Le contrôleur ignore le contrôle |
| `mean_collapse` | `effective_rank` + `intra_batch_sim` des sorties à **contrôle varié** (réutilise les sondes existantes) | rank ↑ / sim ↓ | rank > 0.5·D, sim < 0.5 | rank 0.1–0.5·D, sim 0.5–0.9 | rank < 0.1·D, sim > 0.9 | Régression vers la moyenne |
| `rollout_drift` | Erreur (FK-cosine / L2 vitesses) **vs longueur de rollout** ; horizon court (A1) puis long (A4) | ↓ (plate) | erreur bornée sur l'horizon cible | dérive lente détectable | explosion / freeze | Exposure bias |
| `cold_start_stability` *(ajouté 2026-07-05)* | Rollout **400 frames depuis le seed T-pose identité** (celui des moteurs) : max( déviation d'orthonormalité 6D, \|ΣΔyaw\|/2π, \|ΣΔheight\| m ) | ↓ | < 0.25 | 0.25–1.0 | > 1.0 (tournoiement / envol / démembrement) | Divergence long-horizon depuis pose froide, invisible pour `rollout_drift` (seedé GT, horizon court) — diagnostiquée in-engine 2026-07-05 |
| `style_separation` *(différé A3)* | `z_style` différents → stats de mouvement **mesurablement différentes** | ↑ | séparation nette | marginale | nulle (styles indiscernables) | Style sans effet |
| `style_leakage` *(différé A3)* | À **contrôle fixé**, trajectoire root / timing identiques entre styles | ↓ vers 0 | quasi identiques | divergence modérée | le style change le **quoi** | Fuite du style dans le contenu |

> Les deux contrats `style_*` sont **différés avec la phase A3** : ils ne
> sont câblés que si un dataset labellisé style devient disponible (§1).

Contrats **hérités tels quels** et toujours pertinents : `post_norm_stats`
(assert N(0,1) sur état, deltas **et contrôle `vx,vz`** ; `aim_x,aim_z`
normés-unité exclus — ex-Q4), `nan_inf`, `update_ratio` (‖Δw‖/‖w‖),
`loss_share` (part de chaque composant : L2 vitesses / géodésique 6D /
velocity / foot-contact), `val_gap`, `epoch_time` / `rss_memory`.

Garde-fou anti-usine-à-gaz (identique à ROADMAP §4) : on démarre avec les
**3 contrats nouveaux strictement nécessaires** à A1–A2
(`control_sensitivity`, `mean_collapse`, `rollout_drift`) ; les deux
contrats style (`style_separation`, `style_leakage`) sont **différés avec
A3** (dataset labellisé style requis). Tout ajout ultérieur exige la
justification « quel incident ça aurait détecté ».

> **Flux JSONL branché (2026-07-01).** Les boucles contrôleur (overfit A1,
> généralisation A6) calculent leurs métriques inline (pas de probes
> HealthHub) ; jusqu'ici elles n'écrivaient **aucun** `health.jsonl` — le
> moniteur (`apps/monitor/`) était aveugle sur la voie produit. Corrigé :
> `health/controller_health_writer.py` (`ControllerHealthWriter`) écrit
> `outputDir/health/health.jsonl` au schéma publié (`record_schema`) —
> pertes `loss_*`/`loss_share.*` à chaque `logEvery`, métriques +
> `verdict.*` à l'évaluation finale.

---

## 5. Plan d'exécution Goal A (pour Sonnet)

Même discipline que les autres fiches : chaque phase = un incrément livrable,
testé, **sans casser** `train_generation_v2 --profile {overfit,full}`
(diffusion) ni `lint-imports`. Critères d'acceptation explicites ; ne pas
passer à la phase suivante si l'un d'eux échoue. Le smoke test overfit-1 et
le mode `--debug` s'appliquent au contrôleur.

### Phase A0 — Cadrage de l'état + flags (avant de coder la logique)
- Figer le **vecteur d'état déterministe** (§2.2) dans `core/types`
  (`ControllerState`, `ControlSignal`, `StylePreset`) + indices/noms de
  canaux dans `core/constants`. Schéma Pydantic strict `ControllerV2Config`
  (extra="forbid"), chargé par le loader existant.
- Ajouter le bloc `network.yaml` (§3.2) avec `model-type: diffusion` par
  **défaut** (aucun changement de comportement existant).
- **Acceptation** : un YAML contrôleur avec clé inconnue lève une erreur de
  validation nommant le champ ; tests unitaires du schéma et des
  dataclasses d'état ; `lint-imports` toujours vert ; diffusion inchangée.

### Phase A1 — Contrôleur minimal nu (★ gate de faisabilité)
- `MotionController` autorégressif dans `model/controller_v2.py` :
  `f(state_t, control_t) → Δstate`. **Un seul style, pas de phase encore.**
  Objectif unique : prouver que la **boucle autorégressive s'entraîne et se
  déroule sans exploser**.
- Builder de séquences autorégressives dans `data/` (fenêtres, cibles
  Δstate dérivées de la GT, signal de contrôle dérivé de la GT).
- Loss `losses_controller_v2.py` : L2 sur vitesses + perte **géodésique**
  sur le 6D + `velocity_loss` (flag déjà présent côté v2).
- Boucle de rollout dans `training/` ; contrats `control_sensitivity`,
  `mean_collapse`, `rollout_drift` (horizon court) câblés au HealthHub.
- **Acceptation** : overfit **1 séquence** → rollout court qui **se
  reproduit** (même smoke test qu'en diffusion) ; `rollout_drift` borné sur
  l'horizon court ; `post_norm_stats` OK sur état et deltas ; run `--debug`
  bout-en-bout (train → checkpoint → 1 rollout exporté) < 2 min sur MPS ;
  diffusion + `lint-imports` toujours verts.

### Phase A2 — Tuer la moyenne (phase explicite + contrôle riche)
> **Décision (l'utilisateur, 2026-06-22)** : la **phase explicite est dérivée du
> cycle de marche par contacts pieds** (style PFNN). `deriveFootContacts`
> (hauteur + vitesse planaire du pied, via FK) → `deriveGaitPhase` (chaque
> appui avance la phase de π, interpolation linéaire → `(cos, sin)`).
> Implémenté dans `data/controller_sequences.py`.
>
> **Statut implémentation (2026-06-22)** : A0, A1, A2, A4, A5 livrés et
> testés (code-complete). A3 (style) reste différée (pas de dataset
> labellisé). Restent *gated* par un run réel + validation Blender de
> l'utilisateur : A2 `mean_collapse` SAIN à contrôle varié + réduction
> jitter/sliding ; A4 drift borné sur horizon long.
- Ajouter la **phase en conditionnement** (`phase: explicit`) — l'étape qui
  élimine sliding/jitter. C'est traité comme **non optionnel** (§7).
- Enrichir le **signal de contrôle** : vecteur de **vitesse désirée** +
  **direction de visée** (plus seulement avant/arrière).
- Activer `foot_contact_loss` (supervision FK des contacts via `geometry/`).
- **Acceptation** : `control_sensitivity` nettement > 0 et stable (le
  contrôle module la sortie) ; `mean_collapse` SAIN à contrôle varié
  (effective_rank > 0.5·D, intra_batch_sim < 0.5) ; réduction mesurable du
  jitter/sliding vs A1 (foot-contact + métriques de vélocité) ; validation
  visuelle Blender sur le probe set (walk/jump/run) — responsabilité
  de l'utilisateur.

### Phase A3 — Le style verrouillable (`z_style`) — ⚠ OPTIONNELLE / DIFFÉRÉE
> **Bloquée par les données** : le dataset actuel (AMASS + captions
> HumanML3D/KIT) n'a pas de labels de style (§1). Cette phase n'est lancée
> **que si** un dataset labellisé style devient disponible. Elle n'est PAS
> un prérequis de A4/A5 : la chaîne A1 → A2 → **A4 → A5** se déroule
> entièrement sans elle (`style-latent: false`).
- Injection `z_style` par **FiLM/AdaIN** dans le contrôleur
  (`style_latent.py`, réutilise `layers/`). Entraînement sur le sous-set
  labellisé style ; vecteurs presets stockés dans `configs/styles/`.
  « Lock » = geler `z_style`. Active `style-latent: true`.
- Contrats `style_separation` + `style_leakage` ajoutés au HealthHub.
- **Pré-requis d'entrée** : disposer d'un dataset avec labels de style
  (acquisition/annotation — arbitrage de l'utilisateur), sinon la phase reste parquée.
- **Acceptation** : presets appris et rechargeables depuis `configs/styles/`
  ; `style_separation` SAIN (styles distinguables) ET `style_leakage` SAIN
  (à contrôle fixé, trajectoire root/timing inchangés) ; un ninja avance
  toujours correctement (le style change le comment, pas le quoi).

### Phase A4 — Robustesse longue durée (scheduled sampling)
> Suit directement **A2** (A3 est optionnelle et n'est pas un prérequis).
- Corriger l'**exposure bias** : nourrir le contrôleur de ses **propres
  prédictions** pendant l'entraînement. Flag `scheduled-sampling` **rampé
  de 0 → cible** ; `context-frames` étendu.
- **Acceptation** : `rollout_drift` borné sur **horizon long** (pas de
  freeze/explosion sur un rollout continu équivalent à plusieurs dizaines
  de secondes) ; pas de régression des contrats A2/A3 ; courbe drift vs
  longueur consignée dans `health report`.

### Phase A5 — Export temps réel (la finalité)
- `export/onnx.py` étendu + CLI `export_onnx controller` : exporter **un
  seul forward** du contrôleur (aucune boucle interne). Axes dynamiques
  (batch, frames, longueur de contexte) ; règles §2.10 de ROADMAP
  respectées (pas de control flow data-dépendant, pas de `.item()`).
- Test pytest de parité ONNXRuntime (CPU) vs torch, tolérance 1e-3, marqué
  CI (extension du test d'export existant).
- Intégration moteur **hors scope code** (côté utilisateur) : Unity Sentis /
  Unreal NNE, inférence par frame, état porté en C#/C++, foot-lock IK en
  post + blending physique.
- **Acceptation** : `.onnx` valide (onnx.checker) ; test de parité vert ;
  ops/patterns documentés dans `export/README.md`.

### Phase A6 — Généralisation (entraînement large + eval held-out)
> Décidée par l'utilisateur le 2026-06-23 (abandon de la voie diffusion ;
> le contrôleur devient la voie produit). A0–A5 ont tous été validés en
> **overfit** (1 puis 16 clips) : « le rollout reproduit le clip vu ». A6
> est la première phase qui teste la **généralisation** — répondre à des
> contrôles arbitraires et tenir sur des clips **jamais vus**.
>
> Rappel d'interface (cf. §1) : pour le contrôleur, « donner un prompt » =
> fournir un **signal de contrôle** (`forward:1.0`, direction de visée),
> **pas** du texte. Le texte→contrôle reste un projet distinct (§7).

**Prérequis data — DÉBLOCAGE (vérifié 2026-06-23).** Entraîner sur le set
**upright** `dataset_preprocessed_canon` (Y-up confirmé numériquement : axe
tête→pieds = Y sur 40/40 samples, Y signé ≈ +1.25 ; 13 365 samples, folders
ACCAD/BMLmovi/BioMotionLab). Sur ce set la dérivation foot-contact/gait
phase est **fiable** → entraîner avec `--phase explicit` et
`--foot-contact-weight > 0` (les démos A2 tournaient `phase=none` sur
l'ancien set mis-oriented). **Ne PAS** utiliser `dataset_preprocessed`
(mis-oriented, = défaut implicite du CLI via `preprocess_dataset.yaml`) :
toujours passer `--dataset-root /Users/pazimor/dataset_preprocessed_canon`.

**Trou de code à combler (bloquant l'eval de généralisation).**
- `cli/train_controller_multiclip_v2.py` n'a **aucun split** : il entraîne
  sur tous les clips et mesure `control_sensitivity`/drift sur ces **mêmes**
  clips (train). Ajouter une sélection M clips → **K réservés held-out**
  (jamais vus à l'entraînement, graine déterministe, split caché façon
  `probe_selection.json` de la diffusion).
- Harness d'eval held-out : pour chaque clip réservé, **dériver le contrôle
  de sa GT**, faire le rollout, mesurer (réutilise `controller_rollout` +
  `controller_metrics`, aucun nouveau moteur).

**Protocole d'eval — le juge honnête est le held-out, pas le train.**
1. **Reconstruction held-out** — erreur géodésique/frame (teacher-forced)
   sur clips non vus. Cible : du même ordre que le train (pas d'écart ×10).
2. **Drift closed-loop held-out** — `closedLoopDriftByPeriod` par horizon
   sur clips non vus. Cible : borné, comme A4 sur train.
3. **control_sensitivity held-out** — réponse à un contrôle varié sur clips
   non vus ; cible **> 0.3** (réf. A2 train = 0.544).
4. **`effective_rank` / `intra_batch_sim`** réinterprétés (§7) : détectent
   la **régression vers la moyenne**. Surveiller qu'ils ne s'effondrent pas
   quand N monte (le mal connu de la diffusion).

**Loi capacité↔N (hypothèse héritée de la diffusion, à tester ici).** 256d/4L est
taillé pour ~16 clips en overfit. Pour des centaines/milliers de clips,
monter la capacité (`--embed-dim/--num-heads/--num-layers`) sinon
ré-averaging. Faire un **probe capacité** (N fixe, 2–3 tailles) avant tout
gros run.

**Échelle progressive (pas de saut direct à 13k).**
- Échelon 1 : ~64–128 clips, held-out ~16, capacité 256/4 → baseline.
- Échelon 2 : même N, capacité ↑ → isole le gain capacité.
- Échelon 3 : N×4… jusqu'au plafond mémoire (le CLI charge **tous** les
  clips en RAM ; au-delà → streaming/loading par batch = trou de code
  secondaire à ouvrir le moment venu).
- Scheduled sampling (A4) en **warm-start** (`--resume`) obligatoire pour la
  robustesse rollout à l'échelle.

**Acceptation A6.** Sur held-out : reconstruction du même ordre que le
train, drift closed-loop borné, `control_sensitivity > 0.3` ; courbe
capacité↔N consignée dans `LOG.md` ; aucun contrat A2/A4 régressé.
**Et validation visuelle Blender par Pazimor** (le run 2026-06-26 ci-dessous
montre pourquoi les métriques seules ne suffisent pas).

> **⚠ Run full 2026-06-26 invalidé pour le texte — re-pass A6-bis
> (constat 2026-07-01, cf. LOG).** Le run `controller_text_full` a été
> entraîné via le profil `full` qui n'active PAS le conditionnement texte
> (`promptEmbChannels: 0` dans son `resolved_config.yaml` ; seul le profil
> `controller_text`, variante overfit, l'active) et sans scheduled sampling
> (`scheduledSampling: 0.0`). Conséquence à la génération « walking » :
> prompt **silencieusement ignoré** (le CLI retombe sur l'embedding nul)
> → segment guidé statique ; rollout libre en dérive continue (biais
> d'exposition non mitigé). **Re-pass A6-bis (gated Pazimor —
> G-HYPERPARAMS/G-RUNS, modifs de profil à valider avant lancement) :**
> 1. étendre le profil `full` (ou créer `full_text`) avec
>    `prompt-emb-channels` aligné sur l'artefact encodeur + les champs
>    encodeur du profil `controller_text` ;
> 2. warm-start scheduled sampling > 0 (protocole A4, `--resume`) ;
> 3. ✅ **FAIT (2026-07-01)** — silent-ignore de `generate_controller_v2`
>    corrigé : `_resolvePromptEmb` lève désormais une **erreur explicite**
>    (`SystemExit`) quand un prompt est fourni à un checkpoint sans
>    conditioning (`promptEmbChannels=0`) ou sans `--encoder-artifact`,
>    au lieu du fallback silencieux sur l'embedding nul. Aucun prompt =
>    `None` légitime conservé. Régression verrouillée par
>    `test/ainimator/cli/test_generate_controller_v2.py` ;
> 4. relancer la génération « walking » et valider en Blender.
> Le smoke test overfit reste le gate avant tout run long (G-RUNS) ; les
> runs écrivent désormais `health/health.jsonl` (moniteur, cf. §4).
>
> **Câblages 2026-07-02 (post-diagnostic encodeur, cf. LOG) :**
> l'encodeur `xlm_roberta_artifact` était **dégénéré** (cos inter-prompts
> ≈ 0.998) — artefact canonique remplacé par `output/clip_text_artifact`
> (CLIP text tower, 512) et profils alignés. Le contrat
> **`prompt_sensitivity` est maintenant calculé** à l'évaluation (boucles
> overfit ET généralisation, train + **held-out** — les captions held-out
> sont acheminées par le CLI) au lieu de sortir `UNKNOWN` : l'axe texte
> du re-pass A6-bis est mesurable. Seuils : `health.yaml` (> 0.02 sain).
> Reste `UNKNOWN` uniquement pour un run sans conditionnement texte et
> pour `eval_controller_v2` (pas encore de flag encodeur — trou mineur).

**Risque de fond (l'analogue honnête du problème diffusion).** Le contrôle
dérivé de la GT (vitesse planaire + aim) peut être **trop pauvre pour
désambiguïser** deux motions différentes partageant le même contrôle → le
contrôleur déterministe en **moyennera** (exactement le mal des prompts
dupliqués de la diffusion). Si l'eval held-out montre de l'averaging, le levier
n'est pas la capacité mais l'**enrichissement du signal de contrôle**
(phase, contacts, voire `z_style` A3 si un dataset labellisé apparaît).

### Phase A7 — Cleanup & découplage des CLI (consolidation produit)
> Décidée par l'utilisateur le 2026-06-23, une fois le contrôleur acté comme voie
> produit (cf. A6). Objectif : passer d'une surface « scaffolding
> d'expérimentation » (trois CLI d'entraînement quasi dupliquées, eval
> held-out couplée à la boucle de training) à une surface **propre,
> testable, et prête à alimenter le Goal B** (plugins, cf.
> `doc/ROADMAP_PLUGINS.md`).
>
> **Hors scope (décision de l'utilisateur, 2026-06-23)** : la diffusion
> n'est PAS touchée par cette phase — pas d'archivage du moteur diffusion,
> `model-type` par défaut reste `diffusion` jusqu'à un arbitrage séparé.
> A7 ne nettoie QUE la sous-arborescence contrôleur.

**Constat (état 2026-06-23).** Trois CLI d'entraînement partagent ~90 % de
leur code (mêmes helpers `_resolveDatasetRoot`, sélection de clips, mapping
config) :

| CLI actuelle | Rôle d'origine | Sort en A7 |
|---|---|---|
| `train_controller_v2` | overfit 1 clip (A1) | **fondu** dans le profil `overfit` de la CLI unifiée (reste le smoke test canonique, vérité #7) |
| `train_controller_multiclip_v2` | validation A2/A4 | **archivé** dans `legacy/` (rôle subsumé par le profil `full` à petit N, held-out=0) |
| `train_controller_generalization_v2` | A6, split + held-out | **devient** le profil `full` de la CLI unifiée |

De plus, l'eval held-out vit **dans** `runControllerGeneralization` : on ne
peut pas évaluer un checkpoint arbitraire sans le réentraîner.

**Travail.**
1. **Unifier l'entraînement (config-first)** : une seule CLI
   `train_controller_v2 --profile {overfit,full,debug}` (cf. décision §3.2).
   `overfit` = 1 clip (smoke test), `full` = généralisation (split
   train/held-out), `debug` = bout-en-bout < 2 min. **Tous les anciens flags
   d'hyperparamètre** (sélection `num-clips/held-out-clips/min-frames`,
   archi `embed-dim/num-heads/num-layers`, `phase`, `scheduled-sampling`,
   `dataset-root`) **migrent en champs de profil** dans `network.yaml` — le
   CLI ne les expose plus en flags. Archiver `controller_multiclip_v2`
   (training + CLI) dans `legacy/`.
2. **Découpler l'eval** (le « tester en découplant les CLI » demandé) :
   nouvelle CLI `eval_controller_v2 <checkpoint>` qui charge un checkpoint,
   relit le split déterministe (seed stocké dans `resolved_config.yaml`),
   dérive le contrôle GT des clips held-out, fait le rollout et reporte
   reconstruction / drift closed-loop / `control_sensitivity` — **sans
   réentraîner**. Réutilise `controller_rollout` + `controller_metrics`
   (zéro nouveau moteur).
3. **Hygiène & dead code** : vulture/coverage sur la sous-arbo contrôleur,
   retrait du code mort, factorisation des helpers dupliqués
   (`_resolveDatasetRoot`, `_splitIndices`/`_selectIndices`, `_loadClips`)
   dans un module de la bonne couche (`data/controller_selection.py`),
   docstrings NumPy + Pylance strict + méthodes ≤ 25 lignes + ≤ 80 col.
4. **Formaliser l'artefact contrôleur** : un bundle documenté = poids +
   `resolved_config.yaml` + stats normalizer (état / delta / contrôle `vx,vz` ;
   `aim_x,aim_z` normés-unité, sans stats — ex-Q4).
   C'est le **prérequis du Goal B phase B0** (contrat d'inférence) — le
   format est figé ici et référencé dans `export/README.md`.

**Acceptation.**
- Une seule CLI d'entraînement (`train_controller_v2 --profile
  {overfit,full,debug}`) ; **aucun hyperparamètre exposé en flag** (tout en
  profil config) ; `multiclip` dans `legacy/`, importable nulle part
  (`lint-imports` vert, test négatif ajouté) ;
- `eval_controller_v2 <checkpoint>` reproduit les métriques held-out d'un
  run A6 à partir du seul checkpoint (parité avec les métriques loggées par
  le training, tolérance 1e-4) ;
- smoke test overfit-1 toujours vert via le profil `overfit` ; le profil
  `debug` tourne bout-en-bout < 2 min sur MPS ;
- vulture ne rapporte aucun dead code contrôleur ; `lint-imports` + `pytest`
  verts ; diffusion (`train_generation_v2`) inchangée ;
- bundle artefact contrôleur documenté dans `export/README.md` et produit
  par l'export (lien Goal B / B0).

**Règles pour Sonnet** (rappel) : conventions `doc/GUIDELINES.md` ; commits
atomiques par changement cohérent ; **ne jamais modifier les défauts
d'hyperparamètres existants** sans instruction explicite de l'utilisateur ;
smoke test overfit-1 avant tout run long ; préparer les commandes des runs
> 30 min plutôt que de les lancer en session ; en cas d'ambiguïté, poser la
question plutôt que de choisir silencieusement.

---

## 6. Tensions honnêtes & risques

- **La phase est le seul point où l'on peut encore se planter
  sérieusement.** Ne pas la traiter comme optionnelle. On démarre
  *explicite* (signal fourni) précisément pour dé-risquer A1–A2 ; le
  passage à *apprise* (`phase: learned`) est un arbitrage de l'utilisateur, pas un
  défaut. Le sliding/jitter résiduel est le symptôme à surveiller.
- **Perte de la diversité texte→animation.** Le déterministe régresse un
  mouvement, il n'en échantillonne pas plusieurs. Si la diversité redevient
  un besoin, c'est un projet distinct (distillation maître-élève
  diffusion→contrôleur), pas une rallonge de cette fiche.
- **Style bloqué par les données.** AMASS n'a pas de labels de style, donc
  A3 (`z_style`) est parquée tant qu'un dataset labellisé n'existe pas. Le
  contrôleur reste pleinement utilisable sans : il est piloté par les
  vecteurs de contrôle (vitesse/direction). Ne pas câbler `style_latent`,
  ni les contrats `style_*`, ni `configs/styles/` avant d'avoir ce dataset
  — sinon on entraîne un levier de style sur des données qui ne le
  portent pas (risque direct de fuite du style dans le contenu).
- **Réinterprétation des sondes existantes.** `effective_rank` et
  `intra_batch_sim` détectent désormais l'insensibilité au contrôle / la
  régression vers la moyenne, pas le collapse cond/uncond. Garder cette
  distinction explicite dans `health.yaml` pour ne pas mélanger les
  diagnostics des deux moteurs.
- **Coexistence des deux moteurs.** Tant que `model-type` par défaut reste
  `diffusion`, Goal A est strictement additif. Le jour où le déterministe
  deviendrait le produit, c'est une décision de l'utilisateur qui amenderait §2.1 de
  ROADMAP — elle n'est pas prise ici.

---

## 7. Références

- `doc/ROADMAP.md` — fiche de cadrage du projet (full déterministe).
  Sections réutilisées : §2 (vérités), §3 (couches), §4 (HealthHub),
  §4.1 (table des scores), §5 (rôles).
- `doc/GUIDELINES.md` — conventions de code (paragraphes à ID stable).
- `src/configs/network.yaml` (profil v2) — point d'ajout du flag
  `model-type` + bloc `controller`.
- `src/ainimator/health/` — `probe.py`, `contract.py`, `hub.py`,
  `evaluation.py` (réutilisés ; nouveaux contrats §4).
- `src/ainimator/model/` — `denoiser_v2.py`, `layers/` (FiLM/AdaLN
  réutilisés par le contrôleur), `losses_v2.py` (modèle de
  `losses_controller_v2.py`).
- `doc/experiments/LOG.md` — journal des runs (une ligne par expérience).
- `doc/ROADMAP_PLUGINS.md` — fiche canonique **Goal B** (plugins Unity
  Sentis + Unreal NNE). Consomme l'artefact contrôleur figé en A7 (B0).

---

## 8. Options du tool « health monitor »

Le tool de santé est le CLI unique `ainimator.cli.health` (zéro logique :
il parse les args et délègue à `HealthHub`). Cette section liste ses
options **existantes** (telles qu'implémentées, orientées diffusion) puis
les **extensions Goal A** à ajouter pour le contrôleur. Aucune option
existante n'est modifiée — le déterministe ajoute des flags, il n'en
renomme aucun.

### 8.1 Options existantes (communes aux deux moteurs)

Flag global (avant la sous-commande) : `--log-level
{DEBUG,INFO,WARNING,ERROR}` (défaut `INFO`).

| Sous-commande | Argument / flag | Défaut | Rôle |
|---|---|---|---|
| `watch <runDir>` | `runDir` (positionnel) | — | Suit `runDir/health/health.jsonl` en temps réel (poll 1 s). |
| `audit <checkpoint>` | `checkpoint` (positionnel) | — | Audit hors-ligne d'un checkpoint (adapteur `pytorch-auditor` niveaux 3–4). |
| | `--device` | `cpu` | Device de chargement. |
| | `--output-dir` | dossier du checkpoint | Où écrire le JSON d'audit. |
| `diagnose <checkpoint>` | `checkpoint` (positionnel) | — | Diagnostics de génération hors-ligne. |
| | `--prompt` | `"a person walks forward"` | Prompt de conditionnement (voie diffusion). |
| | `--seeds` | `0,42,123` | Seeds comparées (diversité). |
| | `--cfg-scales` | `1.0,4.0,6.0` | Échelles CFG évaluées (diffusion). |
| | `--frames` | `120` | Longueur de génération. |
| | `--num-steps` | `100` | Steps DDIM (diffusion). |
| | `--device` | `auto` | Device. |
| | `--output-dir` | dossier du checkpoint | Sortie JSON. |
| `report <runDir>` | `runDir` (positionnel) | — | Fiche santé agrégée d'un run. |
| | `--format {markdown,json}` | `markdown` | Format de sortie. |

Runtime knobs dans `src/configs/health.yaml` (câblés dans les boucles
d'entraînement) : `health.enabled` (défaut `true`), `health.everySteps`
(défaut `50` — 1 step sur N).

### 8.2 Extensions Goal A (contrôleur déterministe)

Le mode contrôleur n'a **ni prompt, ni CFG, ni boucle DDIM** : les flags
`--prompt`, `--cfg-scales`, `--num-steps` sont **sans objet**. `diagnose`
doit router selon le `model-type` du checkpoint (lu dans
`resolved_config.yaml`) et exposer, en mode contrôleur, les flags suivants
— à implémenter en A1 (drift court) puis A4 (drift long) :

| Sous-commande | Flag Goal A | Défaut proposé | Rôle |
|---|---|---|---|
| `diagnose <checkpoint>` | `--control` | `"forward:1.0"` | Signal de contrôle (vitesse désirée + direction de visée) sérialisé ; remplace `--prompt`. |
| | `--rollout-frames` | `120` | Longueur du rollout autorégressif (remplace `--frames` ; valeur longue en A4). |
| | `--shuffle-control` | `false` | Évalue `control_sensitivity` : rejoue avec contrôle mélangé, mesure le Δ de sortie. |
| | `--phase-mode {none,explicit,learned}` | lu du checkpoint | Force le mode phase pour le diagnostic. |
| | `--seeds` | `0,42,123` | Réutilisé tel quel (init d'état / bruit d'amorçage). |
| | *(différé A3)* `--styles` | `none` | Liste de presets `z_style` à comparer ; n'a d'effet que si `style-latent: true`. |

Contrats à ajouter dans `health.yaml` (mêmes clés que §4, format identique
aux contrats existants : `direction` / seuils `ok`/`warning`/`critical`) :

- `control_sensitivity` — `direction: higher_is_better`, actif dès A1.
- `mean_collapse` — réutilise les sondes `effective_rank` +
  `intra_batch_sim` de la probe `denoiser_blocks` (re-pointée sur les
  blocs du contrôleur), réinterprétées « à contrôle varié ».
- `rollout_drift` — `direction: lower_is_better`, horizon court (A1) puis
  long (A4) ; nécessite une probe de rollout dédiée (capture l'erreur vs
  longueur, stat scalaire bornée — jamais le tenseur).
- `style_separation` / `style_leakage` — **différés A3**, ajoutés seulement
  avec un dataset labellisé style.

Runtime knobs : `health.enabled` et `health.everySteps` s'appliquent tels
quels à la boucle autorégressive ; en mode `--debug`, garder
`everySteps` bas (santé à chaque step) pour le cycle < 2 min.

> Règle de cohérence : `report` et `watch` ne changent pas — ils lisent le
> JSONL, agnostiques au moteur. Seuls `diagnose` (génération) et la
> sélection de contrats (`health.yaml`) sont sensibles à `model-type`.
