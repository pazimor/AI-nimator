# Contrat d'inférence — bundle contrôleur (B0)

> **Statut** : contrat de la phase B0 (`ROADMAP_PLUGINS.md §3.1/§4`).
> La définition **canonique** du layout I/O (état 136, contrôle 4,
> sortie `Δstate`) vit dans `doc/ROADMAP_DETERMINIST.md §2.2` — ce
> document la **sérialise** pour les moteurs, il ne la redéfinit pas.
> Version de contrat : champ `bundle_version` du manifest (« A7.0 »).

## 1. Le bundle

Produit par `python -m ainimator.cli.export_onnx bundle --checkpoint
<REQUIS> --output-dir <dir>` (le `--checkpoint` est obligatoire, aucun
défaut — vérité §2.9). Contenu :

```
bundle/
├── controller.onnx        # un forward : state+control[+prompt_emb][+phase] → Δstate
├── norm_stats.json        # mean/std : état, delta, contrôle vx,vz
├── manifest.json          # sérialisation du contrat (valide manifest.schema.json)
├── resolved_config.yaml   # provenance du checkpoint (copié si fourni)
└── presets/               # ControlPreset (valident control_preset.schema.json)
    ├── idle.json          #   idle / forward / backward /
    ├── forward.json       #   strafe_left / strafe_right
    └── …                  #   exemples éditables côté moteur
```

Un moteur qui charge un bundle **valide le manifest avant toute
inférence** : version compatible, dims I/O attendues, sinon STOP
(fail-fast, jamais silencieux).

## 2. Entrées / sorties ONNX

Noms **exacts** des tenseurs du graphe (snake_case — vérifiés par
introspection `onnx.load` du bundle de référence) :

| Tenseur | Shape | Présence |
|---|---|---|
| `bone_window` | `(B, context_frames, 22, 6)` | toujours |
| `control` | `(B, control_channels)` | toujours |
| `global_window` | `(B, context_frames, 4)` | toujours |
| `prompt_emb` | `(B, prompt_emb_channels)` | si `prompt_emb_channels > 0` |
| `phase` | `(B, 2)` | si `phase_channels == 2` |
| **sortie** `bone_delta` | `(B, 22, 6)` | toujours |
| **sortie** `global_delta` | `(B, 4)` | toujours |

Aucune boucle dans le graphe : **un forward = une frame**. La boucle
autorégressive, l'intégration `Δstate → état`, le foot-lock IK et le
blending vivent côté moteur (`ROADMAP_PLUGINS.md §3.3`).

## 3. Normalisation côté moteur (norm_stats.json)

1. **État** : z-norm par canal — `(x - state.mean) / state.std`
   (sections `bone_*` pour la fenêtre 22×6, `global_*` pour le root
   local motion 4).
2. **Contrôle** : `vx, vz` z-normés avec `control.mean/std` ;
   `aim_x, aim_z` **normés-unité par construction, PAS de stats**
   (ex-Q4).
3. **Sortie** : le `Δstate` sorti est **normalisé** — dénormaliser avec
   les stats `delta` avant l'intégration.
4. **Phase** : `(cos, sin)` — déjà bornée, pas de stats.
5. **Garantie sur les stats** : tous les `std` sérialisés (état, delta,
   contrôle) sont **plafonnés à `1e-5` au fit, côté Python** — le moteur
   peut diviser directement ; un clamp supplémentaire à l'inférence est
   un no-op autorisé mais inutile. Un bundle avec un std < 1e-5 est
   hors contrat.

## 4. Canal prompt — packaging (arbitrage B0)

**Décision B0 : embeddings pré-calculés, pas de `text_encoder.onnx`
dans le bundle v1.**

- Le runtime consomme `promptEmb` `(B, D)` — jamais du texte. Le calcul
  de l'embedding est **hors boucle par frame** dans tous les cas
  (DETERMINIST §2.4).
- Un `ControlPreset` peut embarquer `prompt_emb` (champ optionnel du
  schéma), pré-calculé à l'authoring via l'encodeur Python (artefact
  encodeur canonique).
- **Pas de prompt actif → le moteur passe le null embedding APPRIS**,
  sérialisé dans `norm_stats.json` (section `prompt.null_emb`). Quand
  `prompt_emb_channels > 0`, l'entrée ONNX `promptEmb` est **requise**
  et un vecteur de zéros n'est PAS un substitut valide (c'est un
  paramètre entraîné, pas une absence).
- Rationale : embarquer l'encodeur texte exigerait le tokenizer BPE en
  C#/C++ (friction forte, valeur runtime nulle — un prompt change à
  l'échelle de l'action, pas de la frame). Un `text_encoder.onnx`
  séparé reste une **extension possible** (bump mineur de
  `bundle_version`), le manifest y est prêt (`prompt_emb_channels`).

## 5. Groupes de conditionnement (bus d'extensibilité)

Actifs : `control`, `phase`, `prompt`. Réservés (déclarés dans
`reserved_input_groups`, vides tant qu'aucun dataset ne les porte) :
`interaction`, `perception`, `reaction`, `morphology`
(DETERMINIST §2.2.d). Règle : toute capacité nouvelle entre par les
**entrées** ; la signature de **sortie ONNX ne change jamais**.

## 6. Presets

Cinq presets locomotion par défaut (`defaultControlPresets`) : `idle`,
`forward`, `backward`, `strafe_left`, `strafe_right` — vitesse
`0.033 m/frame` (≈ 1 m/s @ 30 fps), valeurs **brutes** (le moteur
normalise). Exemples éditables : les jeux définissent les leurs dans
l'éditeur (Unity ScriptableObject / Unreal UDataAsset).

## 7. Bundle de référence (tests de parité moteur)

Générer depuis un checkpoint validé (jamais committé — artefact de
build, vérité §2.9) :

```bash
$AIPY -m ainimator.cli.export_onnx bundle \
  --checkpoint output/controller_full_text_a6ter/checkpoints/controller_overfit_checkpoint.pt \
  --resolved-config output/controller_full_text_a6ter/resolved_config.yaml \
  --output-dir output/reference_bundle
```

La parité Python (torch ↔ ONNXRuntime + step de normalisation moteur,
tolérance 1e-3) est verrouillée par
`test/ainimator/export/test_bundle.py` ; les runtimes Unity/Unreal
doivent reproduire la trajectoire de référence à la tolérance
documentée dans leurs fiches (B1/B2).
