# Design partagé B4 — foot-lock IK + blending (post-traitement moteur)

> **Statut** : design commun de la phase B4 (`ROADMAP_PLUGINS.md §4`).
> Implémentation **native par moteur** (Unity C# / Unreal C++), mais les
> règles, seuils et l'ordre du pipeline sont définis **ici et seulement
> ici** — la parité Unity↔Unreal (vérité §2.7) s'étend au
> post-traitement. Tout écart numérique doit être documenté ici, pas
> improvisé côté moteur.

## 1. Place dans la boucle (rappel §3.3 PLUGINS)

Le post-traitement est **en aval** du contrôleur et **n'entre jamais**
dans la fenêtre autorégressive :

```
dstate → state (intégration)          # runtime B1/B2, inchangé
state  → stateWindow.push(state)      # AVANT tout post-traitement
pose   = footLockIK(state)            # B4 — cosmétique, aval
pose   = blend(pose, engineAnim)      # B4 — idle↔move, physique
applyToSkeleton(pose)
```

Règle dure : **la fenêtre d'état reçoit l'état brut du contrôleur**,
jamais la pose corrigée — sinon le post-traitement contamine
l'autorégression et casse la parité avec le rollout de référence.

## 2. Redérivation des contacts pieds (pas de canal prédit)

Le foot-contact est **loss-only** (DETERMINIST §2.2.a) : le moteur
**redérive** les contacts de la pose, avec exactement la règle du
pipeline data (`data/controller_sequences.py::deriveFootContacts`) :

- Joints pieds SMPL-22 : indices **10 (leftFoot)** et **11 (rightFoot)**
  (FK côté moteur : positions monde des deux pieds).
- Un pied est « en contact » quand, au même frame :
  - `hauteur_monde_pied < 0.05` m (axe Y, Y-up), **ET**
  - `vitesse planaire (XZ) < 0.01` m/frame (déplacement du pied entre
    le frame précédent et le courant).
- Hystérésis moteur (anti-flicker, additif au critère data) :
  entrer en contact exige le critère brut ; en sortir exige de le
  violer pendant **2 frames consécutifs** ou une hauteur > 0.075 m
  (1.5× le seuil). Valeurs par défaut, exposées en propriétés du
  composant.

## 3. Foot-lock IK (two-bone, par jambe)

Par pied verrouillé (contact actif) :

1. **Capture** de la position monde du pied au frame d'entrée en
   contact (`lockedPos`).
2. À chaque frame verrouillé : résoudre un **two-bone IK**
   (hanche→genou→cheville, chaîne SMPL 1→4→7 gauche / 2→5→8 droite,
   effecteur = cheville) pour amener le pied à `lockedPos`. Le knee
   pole vector = direction du genou dans la pose du contrôleur
   (préserve l'orientation naturelle, pas de pole fixe).
3. **Clamp de correction** : si `|lockedPos − posePos| > 0.3` m,
   relâcher le lock (le contrôleur a « décidé » un grand déplacement —
   ne pas étirer la jambe).
4. **Relâchement fondu** : à la sortie du contact, interpoler la
   correction vers zéro sur **0.1 s** (lerp position) — jamais de snap.

Implémentation : two-bone IK analytique standard du moteur
(Unity : solveur manuel ou `AnimationRigging.TwoBoneIKConstraint` ;
Unreal : `FABRIK`/`TwoBoneIK` AnimNode ou solveur manuel équivalent).
Le **résultat numérique** doit suivre les mêmes règles (capture, clamp
0.3 m, fondu 0.1 s) des deux côtés.

## 4. Blending idle↔move (state machine moteur)

- Entrée `move` : `‖(vx, vz)‖ > 0.005` m/frame sur le contrôle **brut**
  demandé (pas le normalisé). Entrée `idle` : en dessous pendant
  **0.25 s**.
- Transition : cross-fade de **0.2 s** entre la pose contrôleur et la
  pose idle du moteur (l'asset idle du jeu, ou la pose contrôleur au
  preset `idle` à défaut).
- Le blending est purement **pose-space, aval** : le contrôleur
  continue de tourner (et de nourrir sa fenêtre) pendant les fondus.

## 5. Acceptation (mesure commune)

- **Foot-sliding avant/après** : déplacement planaire moyen des pieds
  pendant leurs frames de contact, sur une marche `forward` de 10 s
  avec le bundle de référence. Cible : réduction mesurable (> 50 % sur
  la métrique), valeurs consignées dans le README de chaque plugin.
- **Zéro régression de parité runtime** : le test de parité B1/B2 porte
  sur l'état **pré-post-traitement** — inchangé par construction (§1).
- Transition idle↔move sans pop visuel (validation Pazimor).
