# AI-nimator — Fiche de route « Goal C : contrôleur déterministe »

> **Statut** : document de référence (canonical) pour le **Goal C**, un
> *deuxième moteur de génération* déterministe et autorégressif, parallèle
> à la diffusion. Il ne remplace pas `doc/ROADMAP.md` : il le **complète**.
> Toute décision qui contredit ce document doit être tranchée par Pazimor,
> puis reportée ici.
>
> **Lire d'abord** : `doc/ROADMAP.md` (fiche canonique du projet, Goal A
> refactor + Goal B scaling diffusion). Ce document réutilise sans les
> redéfinir : l'architecture en couches (§3.2 de ROADMAP), la surcouche
> santé Probe/Contract/HealthHub (§3.3), la table des scores (§3.5), le
> contrat d'imports `import-linter`, les conventions de code (§2.8) et la
> répartition des rôles (§6).
>
> **Public** : agent d'implémentation (Sonnet) pour le code, agent
> d'expérimentation (Opus) pour le débogage des régressions, Pazimor pour
> les arbitrages.
>
> **Création** : 2026-06-22. Origine : décision Pazimor de tester une voie
> *déterministe temps réel* (cible jeu vidéo, contrôle frame-par-frame)
> en complément de la voie texte→diffusion.

---

## 1. Le pivot en une page

La diffusion (`Goal A`/`Goal B`) génère une animation **complète** depuis
un **prompt texte**, hors-ligne, avec une diversité d'échantillons. Le
**Goal C** vise l'inverse complémentaire : un **contrôleur autorégressif**
qui produit l'animation **une frame à la fois**, piloté par un **signal de
contrôle** (vitesse désirée, direction de visée) plutôt que par du texte,
pour un usage **temps réel** dans un moteur de jeu (Unity Sentis / Unreal
NNE).

Idée centrale : on *régresse* le mouvement au lieu de *l'échantillonner*.
`f(state_t, control_t) → Δstate`. Un seul forward par frame, aucune boucle
de débruitage — ce qui rend l'export ONNX/NPU trivial (cf. §2.10 de
ROADMAP, qui devient ici un acquis et non plus une contrainte délicate).

**Ce que ce pivot fait gagner** : latence temps réel, contrôle interactif,
export NPU simple, foot-lock IK + blending physique côté moteur.

**Ce que ce pivot fait perdre** (assumé) : la diversité texte→animation de
la diffusion. « Génère-moi 50 variantes d'une chute depuis un prompt »
reste du ressort de Goal A/B. Si un jour on veut réinjecter de la
diversité dans le déterministe, le pattern maître-élève (distillation
diffusion→contrôleur) redevient pertinent — **hors scope de cette fiche**.

**Principe de migration** (identique à ROADMAP) : tout reste dans le
package `ainimator`. Le contrôleur est une **variante de la couche
`model/`**, piloté par `src/configs/network.yaml`, gardé par le smoke test
overfit-1, instrumenté par `HealthHub`. Pas de nouveau repo, pas de
réécriture big-bang : squelette d'abord, on valide chaque étape à
l'overfit avant d'avancer.

**Décisions de cadrage actées** (Pazimor 2026-06-22) :

- **Goal C parallèle** : la diffusion reste canonique. Aucune vérité §2 de
  ROADMAP n'est abrogée. Le choix « lequel devient produit » est repoussé
  et tranché expérimentalement, pas par avance.
- **Phase explicite d'abord** : on démarre avec une phase locomotrice
  *fournie en entrée* (signal explicite), puis on ouvre la porte à une
  phase apprise via un flag (`phase: none|explicit|learned`). La phase est
  traitée comme **non optionnelle** — c'est le seul point où l'on peut
  encore se planter sérieusement (cf. §7).
- **Style (`z_style`) différé** (Pazimor 2026-06-22) : le dataset actuel
  (AMASS + captions HumanML3D/KIT) **n'a pas de labels de style**
  exploitables (pas de sous-set ninja/businessman). L'injection de style
  est donc **hors du chemin critique** : flag `style-latent: false` par
  défaut, phase C3 marquée *optionnelle/différée*, conditionnée à
  l'obtention d'un dataset labellisé style. Le conditionnement du
  contrôleur repose, à ce stade, sur les **seuls vecteurs de contrôle**
  (vitesse désirée + direction de visée).

---

## 2. Vérités canoniques du déterministe

Décisions actées propres au Goal C. Elles **s'ajoutent** aux vérités §2 de
ROADMAP ; elles n'en abrogent aucune. Sonnet ne doit PAS les re-débattre ;
si un blocage technique l'exige, escalader à Pazimor.

### 2.1 Ce qui est partagé, hérité, nouveau, ou neutralisé localement

| Vérité ROADMAP §2 | Sort dans Goal C |
|---|---|
| #1 v-pred + cosine + DDIM + Min-SNR | **Neutralisé localement** : pas de schedule ni de sampling dans le contrôleur. Reste pleinement actif côté diffusion. |
| #2 Lean representation 135 ch (rot6d 132 + root_trans 3) | **Hérité et étendu** : c'est le socle de l'état (cf. §2.2). Les signaux FK-dérivables restent supervisés à la loss. |
| #3 Z-normalization obligatoire et assertée | **Hérité tel quel** : post-norm ≈ N(0,1) asserté sur l'état ET sur les deltas. Stats du normalizer embarquées dans le checkpoint. |
| #4 Conditionnement non-bypassable (FiLM/AdaLN/null) | **Réinterprété** : le conditionnement n'est plus le texte mais le *signal de contrôle* + `z_style`. Mêmes leviers (FiLM/AdaLN), même exigence de non-bypass (cf. contrat `control_sensitivity`, §4). |
| #5 CFG dropout si cfgScale > 1 | **Sans objet** : pas de CFG en déterministe. |
| #6 Encodeur texte custom BPE canonique | **Hérité comme dépendance optionnelle** : le contrôleur n'utilise pas de texte par défaut (contrôle = vecteurs). Si un mode texte→style apparaît, il consomme `TextEncoderProtocol` (§2.9 ROADMAP), il ne le redéfinit pas. |
| #7 Smoke test overfit-1-sample | **Hérité, ré-instrumenté** : overfit 1 séquence + rollout qui se reproduit (cf. C1). |
| #8 Conventions de code (§2.8) | **Hérité tel quel** : NumPy docstrings, Pylance strict, pas de magic numbers, méthodes ≤ 25 lignes, fichiers ~500 lignes, dataclasses dans `core/types`, tests isolés. |
| #9 Encodeur texte découplé | **Respecté** : si texte→style, dépendance versionnée, jamais d'import remontant. |
| #10 Exportabilité ONNX (voie NPU) | **Promu d'acquis** : un contrôleur = un seul forward par frame, aucune boucle interne. Les règles §2.10 (pas de control flow data-dépendant, pas de `.item()`, axes dynamiques) restent la loi ; le test d'export CI s'étend au contrôleur (cf. C5). |

### 2.2 Vecteur d'état déterministe (l'invariant à régresser)

C'est l'équivalent du *schema* lean de la diffusion, et son **invariant**.
À figer en C0, avant tout code, dans `core/types` + `core/constants`.

- **Socle hérité (lean v2)** : `rotation6d` (132 ch) + `root_translation`
  (3 ch).
- **Ajouts déterministes** (nécessaires à l'autorégression et au temps
  réel) :
  - **vélocité planaire du root** (déplacement sol par frame),
  - **vitesse angulaire du root** (cap/yaw rate),
  - *(option, flag)* **vitesses articulaires** (lissage des transitions),
  - **labels de contact pied** (foot-contact, pour le foot-lock IK).
- **Phase locomotrice** : *explicite* au départ (signal fourni en entrée),
  flag pour basculer en *apprise*.
- **Style** : `z_style` **différé** — non régressé à ce stade. Le dataset
  actuel n'a pas de labels de style (cf. §1) ; l'état déterministe ne
  réserve donc qu'un **emplacement optionnel** (`z_style` désactivé,
  `style-latent: false`), activé seulement si un dataset labellisé arrive
  (phase C3 optionnelle, §5).

Règle d'or (cohérente avec §2 de ROADMAP) : tout signal **FK-dérivable**
est supervisé **à la loss** via FK (`geometry/`), jamais empilé comme
channel prédit redondant. Le contrôleur prédit `Δstate` ; les contacts et
vélocités servent de cibles de supervision et/ou de conditionnement, pas
de sorties dupliquées.

### 2.3 Le contrôleur est une variante de `model/`, piloté par config

Le moteur de génération est sélectionné par un flag `model.type` dans
`src/configs/network.yaml` (cf. §3.2). Diffusion et contrôleur coexistent ;
`training/` reste l'unique point de jonction `data ↔ model` ; `cli/` reste
sans logique. Aucun nouveau package, aucune entorse au contrat d'imports.

---

## 3. Architecture cible (Goal C)

### 3.1 Placement dans les couches (§3.2 de ROADMAP, inchangé)

Le déterministe ne crée **aucune nouvelle couche**. Il ajoute des fichiers
dans des couches existantes, en respectant « imports vers le bas
uniquement » et « `model` n'importe jamais `data` ».

```
src/ainimator/
├── core/
│   ├── types/        # + ControllerState, ControlSignal, StylePreset
│   │                 #   (dataclasses) ; ControllerV2Config (schéma)
│   └── constants/    # + indices de l'état déterministe, noms de canaux
├── geometry/         # (inchangé) FK réutilisée pour supervision contacts
│                     #   et vélocités ; intégration root_translation
├── data/             # + builder de séquences autorégressives (fenêtres,
│                     #   cibles Δstate, labels contact, signal de phase
│                     #   explicite, signal de contrôle dérivé de la GT)
├── text/             # (inchangé) consommé seulement si texte→style un jour
├── diffusion/        # (inchangé) — neutralisé pour Goal C, intact ailleurs
├── model/
│   ├── controller_v2.py   # ★ NOUVEAU : MotionController autorégressif
│   │                      #   f(state_t, control_t, [phase]) → Δstate.
│   │                      #   FiLM/AdaLN réutilisés (layers/). Emplacement
│   │                      #   z_style optionnel, désactivé par défaut.
│   ├── style_latent.py    # (DIFFÉRÉ C3) injection z_style (FiLM/AdaIN)
│   │                      #   + registre — créé seulement si dataset
│   │                      #   labellisé style ; pas livré en C1–C2.
│   ├── losses_controller_v2.py  # L2 vitesses + géodésique 6D +
│   │                      #   velocity_loss + foot_contact_loss
│   └── layers/            # (réutilise FiLM/AdaLN existants)
├── health/           # + nouveaux contrats (§4), réutilise Probe/Hub
├── training/         # + boucle autorégressive (rollout, scheduled
│                     #   sampling rampé) ; SEUL à voir data ET model
├── export/           # + export ONNX d'un forward contrôleur (C5)
└── cli/              # + train_controller_v2, generate_controller_v2,
                      #   export_onnx controller (sans logique)
configs/
├── network.yaml      # + bloc model.type=controller (flags ci-dessous)
└── styles/           # (DIFFÉRÉ C3) presets z_style — créé seulement si
                      #   un dataset labellisé style devient disponible
```

Le contrat `import-linter` est étendu, pas réécrit : `model/controller_v2`
peut importer `core`, `geometry`, `text`, `diffusion` (comme `denoiser_v2`)
mais **jamais** `data`. Un test négatif est ajouté (sur le modèle du test
A2 : un import remontant casse `lint-imports`).

### 3.2 `network.yaml` — le flag central

Le projet utilise `src/configs/network.yaml` (pas `features.yaml`). On y
ajoute un sélecteur de moteur et un bloc `controller`, en kebab-case comme
le reste du fichier. Défauts choisis = comportement déterministe minimal et
sûr (phase explicite, scheduled sampling à 0). **Aucun défaut
d'hyperparamètre existant n'est modifié** (règle §2.8 / §6).

```yaml
v2:
  generation:
    # Sélecteur de moteur. "diffusion" = comportement actuel (défaut).
    # "controller" = Goal C. Le CLI de training lit ce champ.
    model-type: diffusion          # diffusion | controller

    controller:
      autoregressive: true
      phase: explicit              # none | explicit | learned
      # DIFFÉRÉ : pas de labels de style dans le dataset actuel (§1).
      # Reste false jusqu'à C3 (dataset labellisé style requis).
      style-latent: false
      # Fenêtre de contexte autorégressif (frames vues par forward).
      context-frames: 1            # étendue en C4 (robustesse longue)
      losses:
        velocity-loss: true        # flag déjà présent côté v2
        foot-contact-loss: true
        geodesic-rotation: true
      training:
        # Rampe scheduled sampling 0 → cible (corrige l'exposure bias).
        # Reste à 0 jusqu'à C4 (validation rollout court d'abord).
        scheduled-sampling: 0.0
```

Comme tout run v2, un run contrôleur écrit son `resolved_config.yaml`
(config complète + git SHA + date) dans son `outputDir` (acquis A1).

### 3.3 Registre de styles `configs/styles/` — DIFFÉRÉ (C3)

> ⚠ **Non réalisable avec le dataset actuel.** AMASS + captions
> HumanML3D/KIT n'exposent aucun label de style. Cette sous-section décrit
> la cible *si* un dataset labellisé style devient disponible ; rien ici
> n'est livré en C1–C2, et `style-latent` reste `false`.

`z_style` serait **verrouillable** : « lock » = geler le vecteur de style.
Chaque preset = un fichier versionné (`ninja.yaml`, `businessman.yaml`, …)
référencé par nom dans la config de génération. Le contrôleur apprendrait
les vecteurs sur le sous-set labellisé style. Désentanglement exigé : le
style change le **comment**, jamais le **quoi** (un ninja avance encore
correctement). Vérifié par les contrats `style_separation` /
`style_leakage` (§4), eux aussi différés.

---

## 4. Extensions HealthHub (Goal C)

Les trois objets de la surcouche santé (Probe / Contract / HealthHub,
verdicts `OK / WARNING / CRITICAL`, adapteur `pytorch-auditor`) **ne
bougent pas** (§3.3 de ROADMAP). On ajoute des **contrats spécifiques
déterministe**, déclarés dans `src/configs/health.yaml`, chacun portant en
commentaire sa ligne de la table ci-dessous (direction, cible, seuils),
exactement comme la table §3.5 de ROADMAP.

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
| `rollout_drift` | Erreur (FK-cosine / L2 vitesses) **vs longueur de rollout** ; horizon court (C1) puis long (C4) | ↓ (plate) | erreur bornée sur l'horizon cible | dérive lente détectable | explosion / freeze | Exposure bias |
| `style_separation` *(différé C3)* | `z_style` différents → stats de mouvement **mesurablement différentes** | ↑ | séparation nette | marginale | nulle (styles indiscernables) | Style sans effet |
| `style_leakage` *(différé C3)* | À **contrôle fixé**, trajectoire root / timing identiques entre styles | ↓ vers 0 | quasi identiques | divergence modérée | le style change le **quoi** | Fuite du style dans le contenu |

> Les deux contrats `style_*` sont **différés avec la phase C3** : ils ne
> sont câblés que si un dataset labellisé style devient disponible (§1).

Contrats **hérités tels quels** et toujours pertinents : `post_norm_stats`
(assert N(0,1) sur état ET deltas), `nan_inf`, `update_ratio` (‖Δw‖/‖w‖),
`loss_share` (part de chaque composant : L2 vitesses / géodésique 6D /
velocity / foot-contact), `val_gap`, `epoch_time` / `rss_memory`.

Garde-fou anti-usine-à-gaz (identique à ROADMAP §3.3) : on démarre avec les
**3 contrats nouveaux strictement nécessaires** à C1–C2
(`control_sensitivity`, `mean_collapse`, `rollout_drift`) ; les deux
contrats style (`style_separation`, `style_leakage`) sont **différés avec
C3** (dataset labellisé style requis). Tout ajout ultérieur exige la
justification « quel incident ça aurait détecté ».

---

## 5. Plan d'exécution Goal C (pour Sonnet)

Même discipline que ROADMAP §4 : chaque phase = un incrément livrable,
testé, **sans casser** `train_generation_v2 --profile {overfit,full}`
(diffusion) ni `lint-imports`. Critères d'acceptation explicites ; ne pas
passer à la phase suivante si l'un d'eux échoue. Le smoke test overfit-1 et
le mode `--debug` (acquis A6) s'appliquent au contrôleur.

### Phase C0 — Cadrage de l'état + flags (avant de coder la logique)
- Figer le **vecteur d'état déterministe** (§2.2) dans `core/types`
  (`ControllerState`, `ControlSignal`, `StylePreset`) + indices/noms de
  canaux dans `core/constants`. Schéma Pydantic strict `ControllerV2Config`
  (extra="forbid"), chargé par le loader existant.
- Ajouter le bloc `network.yaml` (§3.2) avec `model-type: diffusion` par
  **défaut** (aucun changement de comportement existant).
- **Acceptation** : un YAML contrôleur avec clé inconnue lève une erreur de
  validation nommant le champ ; tests unitaires du schéma et des
  dataclasses d'état ; `lint-imports` toujours vert ; diffusion inchangée.

### Phase C1 — Contrôleur minimal nu (★ gate de faisabilité)
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

### Phase C2 — Tuer la moyenne (phase explicite + contrôle riche)
> **Décision (Pazimor 2026-06-22)** : la **phase explicite est dérivée du
> cycle de marche par contacts pieds** (style PFNN). `deriveFootContacts`
> (hauteur + vitesse planaire du pied, via FK) → `deriveGaitPhase` (chaque
> appui avance la phase de π, interpolation linéaire → `(cos, sin)`).
> Implémenté dans `data/controller_sequences.py`.
>
> **Statut implémentation (2026-06-22)** : C0, C1, C2, C4, C5 livrés et
> testés (code-complete). C3 (style) reste différée (pas de dataset
> labellisé). Restent *gated* par un run réel + validation Blender de
> Pazimor : C2 `mean_collapse` SAIN à contrôle varié + réduction
> jitter/sliding ; C4 drift borné sur horizon long.
- Ajouter la **phase en conditionnement** (`phase: explicit`) — l'étape qui
  élimine sliding/jitter. C'est traité comme **non optionnel** (§7).
- Enrichir le **signal de contrôle** : vecteur de **vitesse désirée** +
  **direction de visée** (plus seulement avant/arrière).
- Activer `foot_contact_loss` (supervision FK des contacts via `geometry/`).
- **Acceptation** : `control_sensitivity` nettement > 0 et stable (le
  contrôle module la sortie) ; `mean_collapse` SAIN à contrôle varié
  (effective_rank > 0.5·D, intra_batch_sim < 0.5) ; réduction mesurable du
  jitter/sliding vs C1 (foot-contact + métriques de vélocité) ; validation
  visuelle Blender sur le probe set (walk/jump/run) — responsabilité
  Pazimor.

### Phase C3 — Le style verrouillable (`z_style`) — ⚠ OPTIONNELLE / DIFFÉRÉE
> **Bloquée par les données** : le dataset actuel (AMASS + captions
> HumanML3D/KIT) n'a pas de labels de style (§1). Cette phase n'est lancée
> **que si** un dataset labellisé style devient disponible. Elle n'est PAS
> un prérequis de C4/C5 : la chaîne C1 → C2 → **C4 → C5** se déroule
> entièrement sans elle (`style-latent: false`).
- Injection `z_style` par **FiLM/AdaIN** dans le contrôleur
  (`style_latent.py`, réutilise `layers/`). Entraînement sur le sous-set
  labellisé style ; vecteurs presets stockés dans `configs/styles/`.
  « Lock » = geler `z_style`. Active `style-latent: true`.
- Contrats `style_separation` + `style_leakage` ajoutés au HealthHub.
- **Pré-requis d'entrée** : disposer d'un dataset avec labels de style
  (acquisition/annotation — arbitrage Pazimor), sinon la phase reste parquée.
- **Acceptation** : presets appris et rechargeables depuis `configs/styles/`
  ; `style_separation` SAIN (styles distinguables) ET `style_leakage` SAIN
  (à contrôle fixé, trajectoire root/timing inchangés) ; un ninja avance
  toujours correctement (le style change le comment, pas le quoi).

### Phase C4 — Robustesse longue durée (scheduled sampling)
> Suit directement **C2** (C3 est optionnelle et n'est pas un prérequis).
- Corriger l'**exposure bias** : nourrir le contrôleur de ses **propres
  prédictions** pendant l'entraînement. Flag `scheduled-sampling` **rampé
  de 0 → cible** ; `context-frames` étendu.
- **Acceptation** : `rollout_drift` borné sur **horizon long** (pas de
  freeze/explosion sur un rollout continu équivalent à plusieurs dizaines
  de secondes) ; pas de régression des contrats C2/C3 ; courbe drift vs
  longueur consignée dans `health report`.

### Phase C5 — Export temps réel (la finalité)
- `export/onnx.py` étendu + CLI `export_onnx controller` : exporter **un
  seul forward** du contrôleur (aucune boucle interne). Axes dynamiques
  (batch, frames, longueur de contexte) ; règles §2.10 de ROADMAP
  respectées (pas de control flow data-dépendant, pas de `.item()`).
- Test pytest de parité ONNXRuntime (CPU) vs torch, tolérance 1e-3, marqué
  CI (extension du test d'export existant).
- Intégration moteur **hors scope code** (côté Pazimor) : Unity Sentis /
  Unreal NNE, inférence par frame, état porté en C#/C++, foot-lock IK en
  post + blending physique.
- **Acceptation** : `.onnx` valide (onnx.checker) ; test de parité vert ;
  ops/patterns documentés dans `export/README.md`.

**Règles pour Sonnet** (rappel ROADMAP §4) : conventions §2.8 ; commits
atomiques par changement cohérent ; **ne jamais modifier les défauts
d'hyperparamètres existants** sans instruction explicite de Pazimor ;
smoke test overfit-1 avant tout run long ; préparer les commandes des runs
> 30 min plutôt que de les lancer en session ; en cas d'ambiguïté, poser la
question plutôt que de choisir silencieusement.

---

## 6. Répartition des rôles (réutilise ROADMAP §6)

| Acteur | Périmètre Goal C | Interdits |
|---|---|---|
| **Sonnet (implementer)** | Phases C0→C5 dans l'ordre, critères d'acceptation obligatoires ; outillage demandé par l'experimenter (`doc/experiments/requests/`) | Changer les défauts d'hyperparamètres ; toucher aux vérités §2 (ROADMAP et présente fiche) ; lancer des runs > 30 min sans demande |
| **Opus (experimenter)** | Analyse des régressions (mean collapse, drift, leakage), hypothèses, protocole de validation ; dépose une demande dans `doc/experiments/requests/` quand une donnée/un outil manque | Refactor structurel hors `health/` (le signaler via cette fiche) |
| **Sonnet (reviewer)** | Lecture seule : vérifie critères d'acceptation + conventions avant de déclarer une phase terminée | Modifier des fichiers |
| **Pazimor** | Arbitrages (statut diffusion vs déterministe, passage phase explicite→apprise, presets de style, intégration moteur), validation visuelle Blender, mise à jour de cette fiche | — |

**Maintenance** : toute expérience conclue ajoute une ligne dans
`doc/experiments/LOG.md` (date, run, config, métriques, verdict) ; toute
décision nouvelle modifie la section concernée ici, avec date.

---

## 7. Tensions honnêtes & risques

- **La phase est le seul point où l'on peut encore se planter
  sérieusement.** Ne pas la traiter comme optionnelle. On démarre
  *explicite* (signal fourni) précisément pour dé-risquer C1–C2 ; le
  passage à *apprise* (`phase: learned`) est un arbitrage Pazimor, pas un
  défaut. Le sliding/jitter résiduel est le symptôme à surveiller.
- **Perte de la diversité texte→animation.** Le déterministe régresse un
  mouvement, il n'en échantillonne pas plusieurs. Si la diversité redevient
  un besoin, c'est un projet distinct (distillation maître-élève
  diffusion→contrôleur), pas une rallonge de cette fiche.
- **Style bloqué par les données.** AMASS n'a pas de labels de style, donc
  C3 (`z_style`) est parquée tant qu'un dataset labellisé n'existe pas. Le
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
  `diffusion`, Goal C est strictement additif. Le jour où le déterministe
  deviendrait le produit, c'est une décision Pazimor qui amenderait §2.1 de
  ROADMAP — elle n'est pas prise ici.

---

## 8. Références

- `doc/ROADMAP.md` — fiche canonique (Goal A refactor, Goal B scaling
  diffusion). Sections réutilisées : §2 (vérités), §3.2 (couches), §3.3
  (HealthHub), §3.5 (table des scores), §6 (rôles).
- `src/configs/network.yaml` (profil v2) — point d'ajout du flag
  `model-type` + bloc `controller`.
- `src/ainimator/health/` — `probe.py`, `contract.py`, `hub.py`,
  `evaluation.py` (réutilisés ; nouveaux contrats §4).
- `src/ainimator/model/` — `denoiser_v2.py`, `layers/` (FiLM/AdaLN
  réutilisés par le contrôleur), `losses_v2.py` (modèle de
  `losses_controller_v2.py`).
- `doc/experiments/LOG.md` — journal des runs (une ligne par expérience).

---

## 9. Options du tool « health monitor »

Le tool de santé est le CLI unique `ainimator.cli.health` (zéro logique :
il parse les args et délègue à `HealthHub`). Cette section liste ses
options **existantes** (telles qu'implémentées, orientées diffusion) puis
les **extensions Goal C** à ajouter pour le contrôleur. Aucune option
existante n'est modifiée — le déterministe ajoute des flags, il n'en
renomme aucun.

### 9.1 Options existantes (communes aux deux moteurs)

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

### 9.2 Extensions Goal C (contrôleur déterministe)

Le mode contrôleur n'a **ni prompt, ni CFG, ni boucle DDIM** : les flags
`--prompt`, `--cfg-scales`, `--num-steps` sont **sans objet**. `diagnose`
doit router selon le `model-type` du checkpoint (lu dans
`resolved_config.yaml`) et exposer, en mode contrôleur, les flags suivants
— à implémenter en C1 (drift court) puis C4 (drift long) :

| Sous-commande | Flag Goal C | Défaut proposé | Rôle |
|---|---|---|---|
| `diagnose <checkpoint>` | `--control` | `"forward:1.0"` | Signal de contrôle (vitesse désirée + direction de visée) sérialisé ; remplace `--prompt`. |
| | `--rollout-frames` | `120` | Longueur du rollout autorégressif (remplace `--frames` ; valeur longue en C4). |
| | `--shuffle-control` | `false` | Évalue `control_sensitivity` : rejoue avec contrôle mélangé, mesure le Δ de sortie. |
| | `--phase-mode {none,explicit,learned}` | lu du checkpoint | Force le mode phase pour le diagnostic. |
| | `--seeds` | `0,42,123` | Réutilisé tel quel (init d'état / bruit d'amorçage). |
| | *(différé C3)* `--styles` | `none` | Liste de presets `z_style` à comparer ; n'a d'effet que si `style-latent: true`. |

Contrats à ajouter dans `health.yaml` (mêmes clés que §4, format identique
aux contrats existants : `direction` / seuils `ok`/`warning`/`critical`) :

- `control_sensitivity` — `direction: higher_is_better`, actif dès C1.
- `mean_collapse` — réutilise les sondes `effective_rank` +
  `intra_batch_sim` de la probe `denoiser_blocks` (re-pointée sur les
  blocs du contrôleur), réinterprétées « à contrôle varié ».
- `rollout_drift` — `direction: lower_is_better`, horizon court (C1) puis
  long (C4) ; nécessite une probe de rollout dédiée (capture l'erreur vs
  longueur, stat scalaire bornée — jamais le tenseur).
- `style_separation` / `style_leakage` — **différés C3**, ajoutés seulement
  avec un dataset labellisé style.

Runtime knobs : `health.enabled` et `health.everySteps` s'appliquent tels
quels à la boucle autorégressive ; en mode `--debug` (acquis A6), garder
`everySteps` bas (santé à chaque step) pour le cycle < 2 min.

> Règle de cohérence : `report` et `watch` ne changent pas — ils lisent le
> JSONL, agnostiques au moteur. Seuls `diagnose` (génération) et la
> sélection de contrats (`health.yaml`) sont sensibles à `model-type`.
