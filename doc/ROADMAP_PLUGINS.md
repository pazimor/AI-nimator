# AI-nimator — Fiche de route « Goal B : plugins moteur (Unity / Unreal) »

> **Statut** : document de référence (canonical) pour le **Goal B**, la
> mise à disposition du contrôleur déterministe comme **plugin clé-en-main**
> dans deux moteurs de jeu — Unity (Sentis) et Unreal (NNE). Il ne remplace
> ni `doc/ROADMAP.md` ni `doc/ROADMAP_DETERMINIST.md` : il les **prolonge**
> côté intégration. Toute décision qui contredit ce document doit être
> tranchée par Pazimor, puis reportée ici.
>
> **Lire d'abord** : `doc/ROADMAP_DETERMINIST.md` (Goal A — le contrôleur,
> son vecteur d'état, son contrat ONNX, son artefact). Le Goal B **consomme**
> l'artefact contrôleur figé en phase A7 ; il ne définit aucun modèle.
>
> **Public** : Pazimor (intégration moteur C#/C++), agent d'implémentation
> pour l'outillage Python d'export/packaging (côté repo `ainimator`).
>
> **Création** : 2026-06-23. Origine : objectif produit de Pazimor — un
> composant paramétrable qui mappe **un bouton → une action + une animation**
> pilotée par un signal de contrôle, dans Unity et Unreal.

---

## 1. L'objectif en une page

Le Goal A produit un **contrôleur autorégressif** exporté en ONNX : un seul
forward par frame, `f(state_t, control_t, [prompt_emb], [phase]) → Δstate`
(l'embedding prompt est calculé **en amont**, une fois par prompt — pas par
frame — cf. `ROADMAP_DETERMINIST.md §2.4`). Le Goal B
**l'emballe pour un moteur de jeu** : inférence par frame, état porté côté
moteur, foot-lock IK + blending en post, et surtout une **UX d'authoring**
qui permet à un développeur de jeu de brancher une animation sur une action
sans toucher au ML.

**Deux livrables jumeaux** (décision Pazimor 2026-06-23), construits sur un
**contrat d'inférence unique** :

| App | Moteur | Runtime d'inférence | Forme |
|---|---|---|---|
| `unity-sentis` | Unity | **Sentis** (ONNX natif Unity) | package UPM (C#) |
| `unreal-nne` | Unreal | **NNE** (Neural Network Engine) | plugin `.uplugin` (C++/Blueprint) |

Les deux apps **partagent la même spec** (`apps/spec/`) : même artefact
ONNX, même format de stats de normalisation, même schéma de presets de
contrôle. Ce qui change entre elles, c'est uniquement le code moteur (C# vs
C++) et l'intégration éditeur (Inspector vs Details/AnimGraph).

### 1.1 Topo — comment l'utilisateur déclenche une animation (le point qui te bloquait)

Tu imaginais « un champ texte raccordé à l'action qui déplace le
personnage ». C'est la bonne intuition d'UX. La différence diffusion ↔
contrôleur n'est **pas** « texte vs pas de texte » — **les deux acceptent du
texte** — mais **comment** le texte agit :

- **En diffusion**, un prompt (« a person walks forward ») est **échantillonné**
  en une animation entière, hors-ligne. Lourd, non temps réel, pas pilotable
  frame par frame.
- **En contrôleur (ce qu'on exporte)**, le modèle est **tout-en-un** : piloté
  frame par frame par un **vecteur de contrôle** *et* conditionné par un
  **prompt texte** optionnel, dont l'embedding est calculé **en amont** (un
  passage encodeur par prompt, pas par frame). Ce qui disparaît vs diffusion,
  c'est l'**échantillonnage hors-ligne d'un clip entier**, pas le texte.

> **Définition niveau modèle = `ROADMAP_DETERMINIST.md §2.4`** (les deux
> méthodes de guidance : prompt + contrôle ; layout du contrôle §2.2.b ; bus de
> conditionnement §2.2.d). Cette section ne la **redéfinit pas** : elle décrit
> seulement l'**exposition côté plugin** (assets, point d'entrée, bindings).

Côté **plugin**, ces deux entrées modèle sont exposées par **un seul point
d'entrée — l'`Entry`** :

- **Canal contrôle** — le vecteur `(vx, vz, aim_x, aim_z)`, le *où* (locomotion
  + visée, temps réel ; rappel d'usage : « avancer » = `(vz=+1)`, « strafe
  left » = `(vx=-1)`, « reculer » = `(vz=-1)`, visée 2D découplée du
  déplacement — layout figé en DETERMINIST §2.2.b). Alimenté par un
  **ControlPreset nommé** (asset tout fait, simple confort) **ou** câblé
  directement depuis le gameplay (WASD, stick…).
- **Canal prompt** — une **phrase** (champ texte ou asset prompt) → text-encoder
  intégré → embedding, le *quoi* expressif (« danse », « salue », « s'assoit »).

Les deux sont **additifs** (`control` ET/OU `prompt`) : ils entrent ensemble
dans le bloc de conditionnement, donc « danse » + `vz+1` = **une danse qui se
déplace vers l'avant**. Côté contrôle, l'UX de binding reste :

```
[ Input / Bouton ]  →  [ Action ]
        W / ↑              Move : preset "forward"
        S / ↓              Move : preset "backward"
        Q                  Move : preset "strafe_left"
        E                  Move : preset "strafe_right"
       souris             Aim  : direction de visée (continu)
```

Chaque ligne = un **binding** vers un **ControlPreset** (asset versionné). Au
runtime : input pressé → ControlPreset lu → vecteur injecté dans l'ONNX à
chaque frame → `Δstate` appliqué → le perso bouge. Le ControlPreset reste
**facultatif** : c'est une commodité, **pas** une porte d'entrée — on peut
s'en passer et fournir le vecteur directement depuis le code de jeu.

Le **canal prompt** est l'autre porte de l'`Entry`, en parallèle : un champ
texte (ou un asset prompt) → text-encoder intégré → embedding qui conditionne
l'action. Il porte tout ce que le vecteur ne **peut pas** exprimer (actions
expressives : danse, sit, attaque…), cf. `TALK_QUESTIONS.md` Q6.

**Décision de cadrage (Pazimor, 2026-06-24)** : les **deux canaux sont livrés
ensemble**, exposés comme un seul point d'entrée **`Entry`**. Le texte n'est
**plus différé en B6** : il est **requis** dès que l'usage cible inclut des
actions expressives (cinématique, « danse qui avance »…). B6 ne garde que le
**raffinement optionnel** — un mapper « texte libre → vecteur de contrôle »
pour piloter aussi la *locomotion* à la voix —, **pas** l'existence du canal
prompt lui-même.

**Ce que le plugin NE fait pas** (assumé, hérité de Goal A §1/§7) : pas de
diversité « génère-moi 50 variantes » — le contrôleur régresse **un**
mouvement. La diversité reste du ressort de la diffusion, hors
scope ici.

---

## 2. Vérités canoniques du Goal B

Décisions actées propres aux plugins. Elles **s'ajoutent** aux vérités de
`ROADMAP.md §2` et `ROADMAP_DETERMINIST.md §2` ; elles n'en abrogent
aucune.

1. **Le plugin ne définit aucun modèle.** Il consomme l'**artefact
   contrôleur** (bundle figé en A7) : `controller.onnx` + `text_encoder.onnx`
   (canal prompt, modèle tout-en-un) + stats de normalisation +
   `resolved_config.yaml`. Aucune logique d'entraînement ne migre dans le
   moteur.
2. **Un seul forward par frame, aucune boucle interne** (acquis de Goal A
   §2.10/A5). La boucle temps réel vit **dans le moteur** (C#/C++), pas dans
   le graphe ONNX. Le graphe = un step `state+control[+prompt_emb][+phase] →
   Δstate`. Le `prompt_emb` est calculé **hors boucle** (une fois par prompt,
   pas par frame) et ré-injecté tel quel à chaque forward (modèle tout-en-un,
   cf. `ROADMAP_DETERMINIST.md §2.4`).
3. **L'état est porté par le moteur.** Le runtime maintient la fenêtre
   d'état (`context-frames`), applique `Δstate`, ré-injecte. La
   normalisation (état / delta / contrôle `vx,vz` ; `aim_x,aim_z` normés-unité,
   sans stats — ex-Q4) est appliquée **côté moteur** avec les stats du bundle —
   jamais ré-apprise.
4. **Contrat d'inférence unique et versionné** (`apps/spec/`). Les deux
   apps lisent le même contrat ; toute évolution du contrôleur qui change le
   layout d'I/O **bump** la version du contrat et casse volontairement le
   chargement d'un bundle incompatible (fail-fast, pas de silencieux).
5. **Entrées du modèle = un point d'entrée `Entry` à deux canaux
   complémentaires** (cf. §1.1, décision 2026-06-24). Canal **contrôle** =
   vecteur `(vx, vz, aim_x, aim_z)` (le *où*, temps réel ; `ControlPreset` =
   simple confort facultatif, jamais obligatoire) ; canal **prompt** = texte →
   encodeur intégré (le *quoi* expressif). Les deux sont **additifs**
   (`control` ET/OU `prompt`) et entrent **ensemble** dans le conditionnement.
   Le prompt n'est **pas** différé en B6 ; B6 ne garde que le mapper optionnel
   « texte → vecteur de contrôle » (locomotion à la voix).
6. **Foot-lock IK + blending = responsabilité moteur**, en post-traitement
   (acquis Goal A A5). Le contrôleur fournit un mouvement plausible ; le
   moteur verrouille les pieds et mélange avec sa physique / state machine.
7. **Parité Unity ↔ Unreal exigée sur le runtime.** À bundle et contrôle
   identiques, les deux apps produisent la **même trajectoire** (tolérance
   numérique documentée). C'est l'analogue moteur du test de parité
   ONNXRuntime vs torch de A5.
8. **Conventions** : côté Python (export/packaging), celles de `CLAUDE.md`.
   Côté moteur, conventions natives (C# : namespaces + asmdef ; Unreal :
   modules + `.uplugin`), tests de parité obligatoires avant publication.
9. **Bundle frais à chaque build, livré dans le plugin** (décision Pazimor
   2026-06-23). Un plugin ne committe **jamais** un `.onnx` figé : le bundle
   est (re)généré **au moment du build** et copié dans le plugin avant le
   build moteur. Mécanisme retenu : **un orchestrateur à commande unique**
   par moteur (pas de hook natif éditeur). Le checkpoint source est un
   **argument CLI obligatoire** (`--checkpoint`, aucun défaut) — build
   déterministe, zéro bundle périmé, zéro « dernier run » implicite. Le
   bundle généré reste un artefact de build (gitignored), pas un fichier
   versionné.

---

## 3. Architecture cible (Goal B)

> **Vue d'ensemble (étage 3 du système) : `ROADMAP.md` §3.3.** Cette section
> n'est pas une architecture parallèle — elle **détaille** l'étage
> déploiement déjà cadré dans ROADMAP §3.3 : le contrat d'inférence champ par
> champ (§3.1), l'arborescence fichiers (§3.2), la boucle runtime (§3.3) et
> l'orchestrateur de build (§3.4).

### 3.1 Le contrat d'inférence (la clé de voûte)

Tout le Goal B repose sur un **artefact auto-suffisant** produit par le
repo `ainimator` et consommé tel quel par les deux moteurs :

```
artifact bundle (produit par export, figé en A7)
├── controller.onnx          # un forward : state+control[+prompt_emb][+phase] → Δstate
├── text_encoder.onnx        # prompt (tokens) → prompt_emb, calculé hors boucle
│                            #   (modèle tout-en-un, DETERMINIST §2.4 ;
│                            #    packaging exact — onnx séparé vs embeddings
│                            #    pré-calculés — à figer en B0)
├── norm_stats.json          # mean/std : état, delta, contrôle `vx,vz` (+ phase si)
│                            #   — `aim_x,aim_z` normés-unité, PAS de stats (ex-Q4)
├── manifest.json            # version contrat, fps, context-frames,
│                            #   layout des canaux, phaseMode, dims I/O
└── presets/                 # ControlPreset par défaut (forward, backward,
    ├── forward.json         #   strafe_left/right, idle, …) — éditables
    ├── strafe_left.json     #   côté moteur, juste fournis en exemple
    └── …
```

Le `manifest.json` est le **contrat lisible** : noms et tailles exactes des
entrées/sorties ONNX, ordre des canaux de l'état, définition du vecteur de
contrôle, `fps` cible, `context-frames`. Il est la **sérialisation** du
contrat d'I/O, dont la définition canonique (layout état + contrôle, sortie
`Δstate`) vit à **un seul endroit** : `ROADMAP_DETERMINIST.md §2.2` — ne pas
la redéfinir ici. ✅ **Layout figé (2026-06-24, ex-Q1)** : contrôle =
`(vx, vz, aim_x, aim_z)` (4) ; état régressé = `rotation6d` (132) + root
motion local `(Δfwd, Δlat, Δheight, Δyaw)` (4) = **136** ; foot-contact
loss-only (redérivé moteur). Un moteur qui charge un bundle valide ce
manifest avant toute inférence.

**Bus de conditionnement & paramétrage côté plugin.** Le manifest déclare
aussi les **groupes de conditionnement** (cf. `ROADMAP_DETERMINIST.md §2.2.d`)
— actifs (`control`, `phase`, `prompt`) et **réservés** (`interaction`,
`perception`, `reaction`, `morphology`). Pour chaque groupe câblé, le plugin
expose un **système de paramétrage** : le jeu déclare quelles features il
alimente, comment elles **mappent sur les canaux** d'entrée, leurs stats de
normalisation et des **presets** par groupe (au même titre que les
`ControlPreset` de §1.1). Concrètement, l'utilisateur d'`interaction`/
`perception` paramètre, dans l'inspecteur du plugin, la source de chaque
feature (ex. un `Transform` d'arme → canaux `interaction`, un sampler de
heightfield → canaux `perception`) sans toucher au modèle. La sortie ONNX
ne change jamais ; seuls des groupes d'entrée s'activent, avec bump de la
version *d'entrée* du manifest. La personnalisation de **style** se fait
hors bus, par **adaptateur LoRA mergé avant export** (§2.2.e DETERMINIST) —
un bundle = un `.onnx` au style déjà fusionné, donc transparent pour le
plugin.

### 3.2 Structure repo — `apps/`

> **Arborescence : `ROADMAP.md` §3.1** (recap unique de la structure du
> repo, dossier `apps/` détaillé compris).

Le code moteur **vit dans le même repo** (dossier `apps/`, hors du package
`ainimator` — aucune entorse au contrat d'imports Python) : un seul endroit
pour versionner spec + apps + outillage d'export. Le détail fichier par
fichier (spec / build / unity-sentis / unreal-nne) est dans ROADMAP §3.1.

### 3.3 Le runtime moteur (identique des deux côtés, langages différents)

Pseudo-boucle, par frame, exécutée dans le moteur :

```
# --- hors boucle : une fois, au (re)changement de prompt ---
prompt_emb = textEncoder.encode(promptText)   # null embedding si pas de prompt

# --- chaque frame ---
control = activePreset.toVector()        # + aim continu (souris/stick)
x_norm  = normalizer.encode(stateWindow, control, phase)
dstate  = onnx.run(x_norm, prompt_emb)   # un seul forward (prompt_emb réutilisé)
state   = normalizer.decodeDelta(dstate).applyTo(state)
state   = footLockIK(state)              # post : verrouille les pieds
pose    = blendWithPhysics(state)        # post : mélange state machine
stateWindow.push(state)                  # autorégression
applyToSkeleton(pose)
```

La seule chose qui diffère entre Unity et Unreal est l'API d'inférence
(`Sentis Worker.Execute` vs `NNE IModelInstance::RunSync`) et le binding au
squelette (Mecanim/Humanoid vs Skeleton/AnimGraph).

### 3.4 Build & livraison — l'orchestrateur (vérité §2.9)

Le bundle ONNX n'est jamais committé : il est régénéré à chaque build par un
**orchestrateur à commande unique** (`apps/build/`), identique de
structure pour les deux moteurs. Le checkpoint est un argument obligatoire.

> **Makefile.** Le Makefile de confort à la racine (cibles `smoke-test`,
> `debug-run`, `monitor`) a été **supprimé le 2026-06-24** (périmé post-pivot).
> Le **nouveau** `Makefile` est (re)créé **ici, en phase B5**, par
> l'orchestrateur `apps/build/` : il porte les cibles `plugin-unity` /
> `plugin-unreal` ci-dessous (et peut ré-exposer `monitor`/`smoke-test` comme
> raccourcis Poetry). Tant que B5 n'est pas livré, utiliser les commandes
> `poetry run …` directement (cf. `CLAUDE.md`).

```
make plugin-unity   CHECKPOINT=output/controller_gen_n1000/checkpoints/best.pt
make plugin-unreal  CHECKPOINT=output/controller_gen_n1000/checkpoints/best.pt
```

Chaque cible enchaîne, sans état caché :

```
1. export   : python -m ainimator.cli.export_onnx controller --bundle \
                 --checkpoint <REQUIS> --out <tmp_bundle>
              (échoue si --checkpoint absent — pas de défaut, vérité §2.9)
2. validate : le manifest du bundle valide manifest.schema.json ;
              version de contrat compatible avec le plugin cible (sinon STOP)
3. deliver  : copie controller.onnx + text_encoder.onnx + manifest.json +
              norm_stats.json + presets/ dans le dossier ressources du plugin
              cible (unity-sentis/.../Resources ou unreal-nne/Content)
4. build    : Unity  → Unity -batchmode -quit -executeMethod <packTarget>
              Unreal → RunUAT BuildPlugin -Plugin=AInimator.uplugin ...
```

Conséquences : le bundle livré est **toujours** issu du checkpoint passé en
ligne de commande (déterministe, rejouable en CI) ; le dossier ressources du
plugin qui reçoit le bundle est **gitignored** (artefact de build, pas
source) ; les moteurs ne font que **consommer** un bundle déjà posé — ils
n'appellent jamais Python eux-mêmes (pas de hook éditeur). L'orchestrateur
vit dans `apps/build/` (hors package `ainimator`, il invoque seulement la
CLI d'export et les outils moteur).

---

## 4. Plan d'exécution Goal B

Même discipline que les autres fiches : chaque phase = un incrément
livrable, testé, avec critères d'acceptation explicites. **Prérequis
global** : Goal A phase A7 livrée (artefact contrôleur figé) **ET un run A6
validé visuellement en Blender par Pazimor** — précision 2026-07-01 : le run
full du 2026-06-26 est invalidé (texte non conditionné, génération
« walking » statique, cf. `ROADMAP_DETERMINIST.md` A6-bis et LOG) ; on ne
fige aucun artefact pour les plugins tant que le re-pass A6-bis n'a pas
produit un checkpoint validé. Les phases B1 et B2 sont **parallélisables**
une fois B0 figé.

### Phase B0 — Contrat d'inférence + bundle d'export (★ clé de voûte)
*(côté repo `ainimator`, dépend de A7)*
- Écrire `apps/spec/inference_contract.md` + `manifest.schema.json` +
  `control_preset.schema.json` : layout exact des I/O ONNX (contrôle 4 ch
  `(vx,vz,aim_x,aim_z)`, **entrée `prompt_emb`** — calculée hors boucle, cf.
  DETERMINIST §2.4 —, état 136, sortie `Δstate` — cf. DETERMINIST §2.2),
  ordre des canaux, `fps`/`context-frames`, procédure de normalisation côté
  moteur. **Figer aussi le packaging du canal prompt** : `text_encoder.onnx`
  séparé livré dans le bundle **ou** embeddings pré-calculés (arbitrage B0). Le
  schéma déclare aussi les **groupes de conditionnement** (§2.2.d
  DETERMINIST) : actifs vs réservés (`interaction`/`perception`/`reaction`/
  `morphology`), avec mapping features→canaux et stats par groupe (paramétrage
  plugin, §3.1).
- Étendre l'export (CLI, ex. `export_onnx controller --bundle`) pour
  produire le bundle complet (onnx contrôleur + `text_encoder.onnx` +
  norm_stats + manifest + presets par défaut). Le **`--checkpoint` est
  obligatoire** (aucun défaut, aucune auto-détection — vérité §2.9) ; absence
  ⇒ erreur claire. Prévoir une
  **étape optionnelle « merge adaptateur LoRA → export »** (`W' = W + B·A`
  avant l'export ONNX, cf. DETERMINIST §2.2.e) : graphe identique, un style =
  un `.onnx` mergé — non bloquant pour B0, mais le pipeline le réserve.
- Déposer un `reference_bundle/` (petit checkpoint A6) pour les tests de
  parité moteur.
- **Acceptation** : `export_onnx controller --bundle --checkpoint <path>`
  produit un bundle ; sans `--checkpoint`, la CLI échoue avec un message
  explicite ; le manifest valide son schéma ; un test Python charge le
  bundle, fait un forward ONNXRuntime et un step de normalisation, et vérifie
  la parité avec le rollout torch (tolérance 1e-3) ; contrat versionné
  (champ `version`).

### Phase B1 — Runtime Unity (Sentis)
- Package UPM `unity-sentis` : chargement du bundle, `Normalizer` (stats du
  manifest), `StateBuffer` (fenêtre autorégressive), `ControllerRuntime`
  (un forward Sentis par frame), `ControlPreset` (ScriptableObject).
- Scène de démo `Samples~` : une capsule pilotée au clavier (WASD →
  presets), sans IK encore.
- **Acceptation** : la démo tourne en temps réel (≥ 60 fps sur la machine
  cible) ; à contrôle fixe, la trajectoire reproduit le rollout de
  référence du bundle (parité avec le test torch/ONNX de B0, tolérance
  documentée) ; aucun NaN/explosion sur 60 s de rollout continu.

### Phase B2 — Runtime Unreal (NNE)
- Plugin `unreal-nne` : équivalent strict de B1 en C++/NNE
  (`ControllerRuntime`, `Normalizer`, `StateBuffer`, `ControlPreset` =
  `UDataAsset`). Exposition Blueprint minimale (set control, tick).
- Contenu de démo : un pawn piloté au clavier.
- **Acceptation** : mêmes critères que B1 ; **parité Unity ↔ Unreal** : à
  bundle + séquence de contrôle identiques, trajectoires égales (tolérance
  documentée) — c'est le test qui verrouille la vérité §2.7.

### Phase B3 — UX d'authoring (le composant « bouton → action + animation »)
- Le composant que tu décris (§1.1), des deux côtés :
  - **Unity** : un `MonoBehaviour` `AInimatorActionBinder` + un Inspector
    custom (`Editor/`) listant les bindings `[Input] → [Action + ControlPreset]`.
    Le « champ texte » = nom/dropdown de preset. Édition/création de presets
    depuis l'éditeur.
  - **Unreal** : un `UAInimatorActionComponent` + Details panel custom
    (ou nœud AnimGraph), bindings via `UInputAction` → `ControlPreset`.
- Presets par défaut fournis (forward/backward/strafe/idle), entièrement
  éditables.
- **Acceptation** : depuis l'éditeur, sans code, on crée un binding
  touche→preset et on voit le personnage répondre en Play ; presets
  sauvegardés comme assets versionnés ; doc d'authoring (1 page par moteur).

### Phase B4 — Post-traitement : foot-lock IK + blending
- Foot-lock IK (à partir des labels de contact du contrôleur) + blending
  avec la state machine / physique du moteur, des deux côtés. Design partagé
  (`apps/spec/`), implémentation native par moteur.
- **Acceptation** : réduction mesurable du foot-sliding vs B1/B2 (avant/après
  sur une marche) ; transition propre idle↔move ; pas de régression de
  parité runtime (le post-traitement est en aval du contrôleur).

### Phase B5 — Packaging & distribution (orchestrateur de build, §3.4)
- **Orchestrateur** `apps/build/` (cible §3.4) : `build_plugin.py` +
  `Makefile` (`plugin-unity` / `plugin-unreal`) qui enchaîne export bundle
  (`--checkpoint` requis) → validation manifest/version → copie dans le
  plugin → build moteur. Le dossier ressources qui reçoit le bundle est
  gitignored.
- Unity : package UPM versionné (git URL / tarball) + scène de démo +
  README ; dépendance Sentis déclarée. Build via `Unity -batchmode`.
- Unreal : `.uplugin` empaquetable (Marketplace-ready ou drop-in) + projet
  de démo + README ; dépendance NNE déclarée. Build via `RunUAT BuildPlugin`.
- Versionnage aligné sur la version du **contrat d'inférence** (un plugin
  déclare quelles versions de bundle il accepte).
- **Acceptation** : `make plugin-unity CHECKPOINT=<path>` et `make
  plugin-unreal CHECKPOINT=<path>` produisent, depuis zéro, un plugin
  packagé contenant un bundle **frais** issu du checkpoint donné ; sans
  `CHECKPOINT`, la cible échoue clairement ; installation propre dans un
  projet vierge (Unity et Unreal) en suivant le seul README ; la démo
  tourne ; le plugin refuse proprement un bundle de version de contrat
  incompatible.

### Phase B6 — Texte → contrôle (OPTIONNELLE / DIFFÉRÉE)
> ⚠ **Ne concerne PAS le canal prompt** (qui est livré par défaut, modèle
> tout-en-un — cf. §1.1 / DETERMINIST §2.4). B6 n'ajoute qu'un **raffinement** :
> piloter aussi la *locomotion* via du texte libre, en mappant une phrase sur un
> **vecteur de contrôle**. Lancée seulement si Pazimor le décide ; **pas un
> prérequis** de B1–B5.
- Un mapper léger texte → vecteur de contrôle : table de synonymes /
  embedding qui résout une phrase vers le preset le plus proche (ou interpole
  un contrôle). Peut vivre côté moteur (offline) ou comme petit ONNX séparé.
- **Acceptation** : un champ texte libre (« cours vers la gauche ») résout un
  contrôle plausible ; comportement déterministe et documenté ; aucun impact
  sur le chemin presets (qui reste le défaut).

### Phase B7 — Encodage de prompt in-engine (lancée 2026-07-04)
> Lève l'arbitrage B0 « embeddings pré-calculés seulement » en activant
> l'**extension prévue** par `inference_contract.md §4` : `text_encoder.onnx`
> + `tokenizer/` (vocab/merges CLIP verbatim) dans le bundle, bump **mineur**
> `A7.0 → A7.1`. Design commun : **`apps/spec/text_encoding.md`** ;
> tokenizer normatif : `apps/spec/clip_bpe_reference.py` + parité
> `apps/spec/text_encoding_parity.json` (même patron que B6).
- Export : `export_onnx bundle --encoder-artifact <dir>` (aussi
  `make plugin-{unity,unreal} … ENCODER_ARTIFACT=output/clip_text_artifact`).
  Graphe une passe `input_ids + attention_mask → prompt_emb` (pooling
  masked-mean **dans** le graphe), axe batch dynamique, T figé à 32.
- Moteurs : `ClipBpeTokenizer` (C# / C++) mirroir valeur-pour-valeur de la
  référence + `PromptTextEncoder` (Sentis / NNE) ; API
  `SetPromptText(string) → bool` qui débouche sur le `SetPromptEmbedding`
  existant (cross-fade, null_emb au repos). Distinct de B6
  (`SetTextCommand`, bas-niveau) — les deux coexistent.
- **Acceptation** : cf. `text_encoding.md §4` — bundle A7.1 valide (schéma) ;
  parité tokenizer 3-voies (pytest / EditMode / Automation, 12 cas
  canoniques) ; parité pooled torch↔ORT ≤ 1e-3 ; `SetPromptText` échoue
  proprement sur un bundle A7.0 ; zéro régression presets.

### Phase B8 — Boucle visuelle & itération outillée (décidée 2026-07-06)
*(élaborée avec Pazimor ; ordre d'exécution V1 → V2 → G1 → G2 → U1/UE1 → U2.
Les fiches V*/G* vivent côté repo `ainimator`, les U*/UE* côté plugins.)*

- **V1 — `render_rollout` (visualisation sans Blender)** : à partir d'un
  rollout sauvé (`generate_controller_v2 --output`), produire (a) un
  contact-sheet PNG (keyframes multi-vues + trajectoire racine, rendu PIL —
  pas de dépendance nouvelle) et (b) un viewer HTML autonome (canvas,
  play/pause/scrub, animation embarquée). Logique dans
  `health/` (outil de debug), CLI mince.
  *Acceptation* : une commande transforme un `.pt` de rollout en PNG + HTML ;
  un agent peut diagnostiquer visuellement un rollout en lisant le PNG.
- **V2 — boucle overfit une-commande** : enchaîner train overfit →
  generate → render en une commande (interpréteur venv direct tant que
  poetry est cassé). *Acceptation* : une commande, < 10 min, sortie
  visuelle V1 en bout de chaîne.
- **G1 — contrôle synthétique scriptable** : piste de contrôle définie par
  fichier (segments vitesse/direction/prompt, l'équivalent d'une séquence
  clavier) en alternative au contrôle dérivé du GT dans
  `generate_controller_v2` — c'est le vrai test de parité avec le moteur.
  *Acceptation* : un rollout piloté 100 % sans dataset (hors état initial).
- **G2 — post-process correcteur (sortie de rollout)** : module Python
  déterministe grounder (projection sol / anti-pénétration) + foot-lock par
  détection de contact (miroir du B4 moteur) + lissage existant. **Hors
  graphe ONNX** (post-process pur) ; les métriques `health/` se mesurent
  sur le rollout **brut** (le correcteur ne doit pas maquiller les
  régressions). *Acceptation* : rendu brut vs corrigé côte à côte ;
  contacts pieds stables sur un cycle de marche corrigé.
- **U1 — prefab Unity tout-en-Inspector** : prefab posé en scène (pas
  d'instanciation runtime) avec bonhomme bâton SMPL-22 procédural par
  défaut, **slot `Model` drag-and-drop** (si renseigné : RigMap/retarget,
  bâton masqué), bindings WASD par défaut pré-câblés via
  `AInimatorActionBinder` (éditables Inspector, zéro code), prompt piloté
  par script (`SetPromptText`/`SetPrompt`) + champ prompt initial.
  *Acceptation* : drag du prefab + bundle → personnage animé au WASD en
  Play mode sans écrire une ligne de code.
- **UE1 — parité Unreal** : même surface que U1 sur `apps/unreal-nne/`
  (le miroir existe déjà : `AInimatorDemoPawn`, `AInimatorActionBinding`,
  `AInimatorSmpl22Skeleton`) : pawn/Blueprint prêt-à-poser, bâton
  procédural, slot mesh, bindings par défaut Enhanced Input éditables en
  Details panel, prompt par Blueprint/C++. Le **bundle est partagé tel
  quel** (même ONNX + manifest) ; quand G2 fige un correcteur, il se
  réplique dans les deux runtimes (comme le fix §3.6). *Acceptation* :
  parité fonctionnelle avec U1, checklist commune.
- **U2 — recette host project (Pazimor)** : prefab U1 + rig Mixamo dans le
  projet de test Unity ; checklist de recette visuelle. *Acceptation* :
  verdict Pazimor consigné dans LOG.md.

---

## 5. Répartition des rôles

| Acteur | Périmètre Goal B | Interdits |
|---|---|---|
| **Agent implémenter** (repo `ainimator`) | B0 : contrat + export du bundle + tests de parité Python/ONNX ; outillage demandé pour le packaging | Toucher aux défauts d'hyperparamètres ou aux vérités §2 des autres fiches ; écrire du code moteur sans validation Pazimor |
| **Pazimor** | Intégration moteur (C#/C++), phases B1–B5, validation visuelle en jeu, arbitrages (texte→contrôle B6, foot-lock, packaging), maintenance de cette fiche | — |
| **Agent reviewer** | Lecture seule : critères d'acceptation B0 (côté Python) + cohérence du contrat | Modifier des fichiers |

**Maintenance** : toute étape conclue ajoute une ligne dans
`doc/experiments/LOG.md` (date, app, version de contrat, verdict) ; toute
décision nouvelle modifie la section concernée ici, avec date.

---

## 6. Tensions honnêtes & risques

- **Le contrat d'inférence est le point de rupture.** Si le layout d'I/O du
  contrôleur change après B0 sans bump de version, les deux plugins cassent
  silencieusement. D'où le `version` obligatoire + fail-fast (§2.4).
- **Parité Unity ↔ Unreal non triviale.** Sentis et NNE peuvent différer sur
  l'ordre des ops / la précision. La tolérance doit être mesurée et
  documentée, pas supposée (test B2).
- **Qualité héritée du contrôleur.** Le plugin n'améliore pas le mouvement :
  si A6 averageait (cf. risque de fond A6), le plugin le montrera tel quel.
  Le foot-lock IK (B4) masque le sliding, pas l'averaging. La qualité se
  gagne en amont (Goal A), pas ici.
- **Le texte → *locomotion* est un piège.** Le **canal prompt** (intention
  haut-niveau) est, lui, un acquis du modèle tout-en-un. Le piège est de mapper
  du **texte libre vers le vecteur de contrôle** (B6) : ça réintroduit dans la
  *locomotion* l'ambiguïté que le contrôle vectoriel évacuait. Garder ce mapper
  optionnel et en aval protège le chemin produit ; le canal prompt, lui, reste
  le chemin normal des actions expressives.
- **Pas de diversité.** Si un besoin « plusieurs variantes » émerge, c'est un
  projet distinct (distillation diffusion→contrôleur, cf. Goal A §7), pas une
  rallonge de ce plugin.

---

## 7. Références

- `doc/ROADMAP_DETERMINIST.md` — Goal A : contrôleur, vecteur d'état (§2.2),
  export ONNX A5, **artefact figé en A7** (prérequis B0).
- `doc/ROADMAP.md` — fiche canonique projet (couches, conventions §2.8,
  vérité ONNX §2.10).
- `src/ainimator/export/` — `onnx.py`, `README.md` (point d'extension pour
  le bundle B0).
- `src/ainimator/model/controller_rollout.py`,
  `src/ainimator/health/controller_metrics.py` — réutilisés par les tests de
  parité B0.
- `doc/experiments/LOG.md` — journal (une ligne par étape conclue).
