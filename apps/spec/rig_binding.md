# Design partagé — binding rig + pilotage par prompt (B3-bis)

> **Statut** : design commun Unity/Unreal de l'étage « le composant
> s'applique sur un personnage riggé et le pilote ». Étend B3/B4 ;
> mêmes règles de parité que `footlock_blending.md` : les maths de
> retargeting et la sémantique prompt sont définies **ici et seulement
> ici**, implémentation native par moteur.
>
> **Vision produit (Pazimor, 2026-07-04)** : le modèle tourne en
> continu ; on lui fournit des **prompts** (haut niveau — quel
> mouvement) et des **vecteurs de direction** (bas niveau) ; changer
> **seulement le prompt** doit suffire à gérer l'animation. Le
> composant s'attache à un personnage riggé et le pilote.

## 1. Chaîne complète (par frame)

```
prompt_emb (actif)  ──┐            # change à l'échelle de l'action
control (vx,vz,aim) ──┤
                      ▼
        ControllerRuntime (B1/B2, inchangé)   → state SMPL-22 brut
                      ▼
        StateBuffer.push(state brut)          # règle dure B4 §1
                      ▼
        PostProcess (B4 : foot-lock, blend)   → pose SMPL-22 corrigée
                      ▼
        ★ RigBinder (CE document)             → squelette du personnage
                      ▼
        root motion → transform de l'acteur
```

## 2. Retargeting SMPL-22 → rig arbitraire (canonique, parité)

Le modèle sort des **rotations locales** SMPL-22 (+ root motion). Un
rig réel (Mecanim Humanoid, UE Mannequin, rig custom) a d'autres os,
d'autres noms, d'autres poses de repos. Voie **canonique commune**
(celle que le test de parité couvre) :

### 2.1 Bone map explicite (asset par rig)

Un asset `RigMap` (ScriptableObject / UDataAsset) : pour chacun des 22
os SMPL (ordre canonique de `core/constants/skeletons.py`, déjà porté
en constantes B4), le nom/référence de l'os du rig — ou « non mappé »
(os SMPL sans équivalent : la rotation est composée avec l'enfant
suivant mappé... **non** : v1 = os non mappé → rotation ignorée,
documenté ; les 22 os SMPL couvrent les humanoïdes standards).

### 2.2 Correction de pose de repos (calibration au bind)

Au bind (une fois, à l'attache du composant) :

1. Poser le rig dans sa **pose de repos** (celle de son asset).
2. Pour chaque os mappé, calculer l'offset entre l'orientation de
   repos SMPL (identité sur le squelette canonique B4) et
   l'orientation de repos **monde** de l'os du rig :
   `O_self = worldRot_rig_rest(bone)`.
3. À chaque frame, la rotation locale à appliquer au rig est :

```
R_rig_local(bone) = O_parent⁻¹ · R_smpl_world(bone) · O_self_localBasis
```

où `R_smpl_world` est la rotation **monde** SMPL obtenue par la FK B4
(composition des locales le long de la hiérarchie), `O_parent` l'offset
de repos du parent mappé. Concrètement : on transporte l'orientation
monde SMPL dans le repère de repos du rig — la formule exacte,
identique aux deux moteurs, est :

```
worldRot_target(bone) = R_smpl_world(bone) · worldRot_rig_rest(bone)
localRot_target(bone) = worldRot_target(parent)⁻¹ · worldRot_target(bone)
```

(`parent` = parent **mappé** le plus proche dans le rig.) Les
translations d'os ne sont **jamais** modifiées (proportions du rig
préservées) ; seule la racine reçoit la root motion.

### 2.3 Échelle et root motion

- **Échelle** : facteur `rigScale` = hauteur pelvis-repos du rig /
  hauteur pelvis SMPL (~0.91 m sur le squelette canonique). La root
  motion (Δfwd, Δlat, Δheight) est multipliée par `rigScale` avant
  application au transform de l'acteur. Δyaw est sans dimension.
- Le foot-lock B4 opère **avant** retarget, dans l'espace SMPL —
  ses positions corrigées passent par le même chemin.

### 2.4 Voie moteur-native (optionnelle, non canonique)

Chaque plugin PEUT offrir en plus la voie native (Unity Humanoid via
`HumanPoseHandler`, UE via IK Retargeter). Elle n'est **pas** couverte
par la parité cross-moteur ; le RigMap explicite reste le chemin de
référence et le défaut.

## 3. Sémantique du prompt à chaud

- API du composant (identique deux moteurs) :
  - `SetPrompt(preset)` — utilise le `prompt_emb` pré-calculé du
    preset (contrat B0 §4) ;
  - `SetPromptEmbedding(float[])` — vecteur brut (D =
    `prompt_emb_channels`) fourni par le jeu ;
  - `ClearPrompt()` — retour au `null_emb` appris (jamais des zéros).
- **Changement de prompt = cross-fade dans l'espace d'embedding** :
  lerp linéaire `emb_old → emb_new` sur **0.3 s** (défaut, exposé).
  ⚠ Heuristique : le modèle n'a jamais vu d'interpolation
  d'embeddings à l'entraînement — comportement à valider visuellement ;
  si transitions dégradées, repli documenté = switch sec + cross-fade
  de POSE (mécanisme B4 §4 réutilisé).
- Le contrôle (vx, vz, aim) reste appliqué chaque frame,
  indépendamment du prompt — les deux entrées sont additives
  (DETERMINIST §2.4).

## 4. Texte libre à l'exécution — HORS bundle v1 (rappel B0)

Le runtime consomme des **embeddings**, pas du texte (B0 §4). Pour
« donner un prompt » :

1. **Authoring** (chemin par défaut) : encoder la phrase côté Python —
   CLI `ainimator.cli.encode_prompt` → JSON `prompt_emb` à coller dans
   un preset (ou charger à chaud via `SetPromptEmbedding`).
2. **Texte libre in-game** : nécessite l'encodeur dans le moteur
   (`text_encoder.onnx` + tokenizer CLIP en C#/C++) — extension B0
   prévue mais **non livrée** ; arbitrage Pazimor si le besoin devient
   réel.

## 5. Acceptation

- Un personnage riggé (humanoïde standard) piloté en Play/PIE par le
  composant : marche upright, membres cohérents, pas de twist d'os
  aberrant (validation visuelle Pazimor).
- `SetPrompt`/`SetPromptEmbedding` à chaud : le mouvement change sans
  crash ni NaN ; qualité de transition consignée (cross-fade embedding
  vs switch sec).
- Parité : à RigMap équivalent et même bundle, `localRot_target` égaux
  Unity/Unreal (tolérance numérique documentée).
- Le chemin capsule (B1/B2) reste fonctionnel (le RigBinder est un
  étage optionnel).
