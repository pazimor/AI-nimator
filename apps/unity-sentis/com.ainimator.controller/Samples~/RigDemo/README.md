# Rig Demo (B3-bis) — personnage riggé piloté par le contrôleur

Démo drag & drop de la phase **B3-bis** (`apps/spec/rig_binding.md`) : un
mannequin SMPL-22 (22 bones + visuels stick-figure, généré procéduralement au
Play) piloté par `AInimatorCharacter` → `RigBinder` → `RigMap`, avec swap de
prompt à chaud (cross-fade d'embedding), prompt texte libre (B7) et commandes
texte (B6).

## Installation dans Unity (pas à pas)

Prérequis : **Unity 6 / 2023.2+** (le package tire `com.unity.ai.inference`
2.6.1 — Sentis — automatiquement).

1. **Installer le package** : `Window > Package Manager` → `+` →
   **Add package from disk...** → sélectionner
   `apps/unity-sentis/com.ainimator.controller/package.json`.
2. **Livrer le bundle contrôleur** (le runtime le lit dans
   `Application.streamingAssetsPath`) : copier le dossier
   `apps/unity-sentis/com.ainimator.controller/StreamingAssets/AInimatorBundle/`
   vers **`Assets/StreamingAssets/AInimatorBundle/`** de votre projet.
   *(Alternative sans copie : renseigner `Bundle Directory Override` sur le
   prefab avec le chemin absolu du dossier bundle.)*
   L'éditeur convertit ensuite automatiquement `controller.onnx` (et
   `text_encoder.onnx`) en `.sentis` — le seul format que Sentis 2.x charge
   au runtime. Si la conversion n'a pas tourné (pas de recompilation depuis
   la copie), forcer via le menu **AInimator > Convert Bundle ONNX to
   Sentis (StreamingAssets)**. ⚠ La conversion automatique ne couvre que le
   bundle StreamingAssets — avec `Bundle Directory Override`, appeler
   `BundleSentisConverter.ConvertBundleIfNeeded(<chemin>)` ou déposer un
   `.sentis` soi-même.
3. **Importer le sample** : Package Manager → *AI-nimator Controller
   (Sentis)* → onglet **Samples** → **Rig Demo (B3-bis)** → *Import*.
   Les fichiers arrivent sous `Assets/Samples/AI-nimator Controller (Sentis)/<version>/Rig Demo (B3-bis)/`.
4. **Glisser `RigDemo.prefab` dans une scène vide** (File > New Scene suffit)
   et appuyer sur **Play**. Le prefab construit tout seul : mannequin riggé,
   sol, lumière, caméra de suivi et HUD.

> ⚠ **Le prefab est VIDE en mode édition — c'est normal.** Le mannequin
> (bones + visuels), le sol et la caméra sont générés **au Play** par
> `RigDemoBootstrap.Awake()` (aucun asset FBX/scène binaire ne peut être
> authoré hors d'un éditeur Unity, même choix que `CapsuleDemoBootstrap`).
> Si au Play il ne se passe toujours rien : ouvrir la **Console** — l'erreur
> quasi certaine est le bundle absent de `Assets/StreamingAssets/AInimatorBundle/`
> (étape 2).

## Utiliser votre propre personnage (Mixamo / FBX humanoïde)

1. Importer le FBX dans le projet et glisser une instance dans la scène,
   **dans sa pose de repos** (T-pose/A-pose d'import), pieds au sol (y=0).
2. Sélectionner le prefab `RigDemo` dans la scène et assigner l'instance du
   personnage au champ **Custom Rig Root**.
3. Play. Au démarrage, les 22 bones SMPL sont auto-mappés par nom
   (conventions Mixamo `mixamorig:Hips`..., Humanoid FBX, Mannequin — cf.
   `RigMapAutoMapping`), l'`Animator` du FBX est désactivé (sinon il
   écraserait les rotations), et le personnage est piloté par le contrôleur
   via `RigBinder`. La Console log le compte de bones mappés et liste les
   bones ignorés.

Si l'auto-map rate des bones (noms exotiques) : créer un asset
`AInimator > Rig Map`, utiliser le bouton **Auto-Map From Root** de son
Inspector puis compléter à la main, et monter `RigBinder` +
`AInimatorCharacter` soi-même (cf. `docs/rig_binding.md` du package).

## Contrôles en Play

| Entrée | Effet |
|---|---|
| `W`/`Z`/`↑` (maintenu) | preset `forward` |
| `S`/`↓` | preset `backward` |
| `A`/`Q`/`←` | preset `strafe_left` |
| `D`/`→` | preset `strafe_right` |
| (aucune touche) | preset `idle` |
| `1`..`9` | `SetPrompt(preset)` — cross-fade vers l'embedding pré-calculé du preset (la liste affichée dépend des presets du bundle qui embarquent un `prompt_emb`) |
| `0` | `ClearPrompt()` — retour au null embedding appris |
| Champ *Free-text prompt* + **Set** | B7 : encode la phrase avec l'encodeur texte du bundle (`text_encoder.onnx` + tokenizer) et cross-fade vers l'embedding |
| Champ *Text command* + **Run** / **Stop** | B6 : résout la phrase (« avance », « walk forward »...) en vecteur de contrôle ; une touche maintenue reprend la priorité |

## Ce que valide cette démo (acceptation B3-bis, spec §5)

- Personnage riggé humanoïde piloté en Play : marche upright, membres
  cohérents, pas de twist aberrant (validation visuelle).
- `SetPrompt` / `SetPromptEmbedding` à chaud : le mouvement change sans crash
  ni NaN ; qualité du cross-fade d'embedding à consigner.
- Le chemin capsule (B1) reste intact — le RigBinder est un étage optionnel.

## Notes / limites connues

- Le mannequin par défaut est un stick-figure 1:1 avec le squelette SMPL-22
  (rigScale = 1) ; le champ **Custom Rig Root** (section ci-dessus) branche
  la même chaîne sur un vrai personnage, avec rigScale calculé depuis la
  hauteur du pelvis.
- Taper dans les champs texte n'inhibe pas les touches WASD (le runtime lit
  `Input.GetKey` globalement) : relâcher les touches de déplacement avant de
  taper.
- La **qualité du mouvement dépend du bundle** livré dans StreamingAssets
  (checkpoint d'entraînement) — un bundle issu d'un run non validé peut
  produire des poses dégradées : c'est le modèle, pas le binding.
- Android/WebGL non couverts : `BundleLoader` lit StreamingAssets par le
  filesystem (cible desktop/éditeur, cf. `BundlePaths`).
