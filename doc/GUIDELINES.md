# AI-nimator — Guidelines de code

> Chaque guideline est un **paragraphe auto-suffisant** identifié par un **ID
> stable** (`G-XXX`) placé sur sa ligne de titre. Pour récupérer une seule
> guideline sans lire tout le fichier :
>
> ```bash
> grep -A8 'G-DOCSTRING' doc/GUIDELINES.md     # ou: rg -A8 'G-DOCSTRING'
> ```
>
> L'index ci-dessous liste tous les IDs. Les IDs ne changent jamais (on en
> ajoute, on n'en renomme pas) — ils peuvent être cités depuis le code, les
> revues, ou les autres fiches.

## Index

- `G-DOCSTRING` — docstrings NumPy
- `G-TYPES` — typage strict (Pylance)
- `G-FORMAT` — formatage (black / isort / ruff, PEP 8)
- `G-MAGIC` — pas de magic numbers / strings
- `G-PURE` — fonctions explicites, pures, réutilisables
- `G-NAMING` — pas de noms raccourcis
- `G-METHODLEN` — méthodes ≤ 25 lignes
- `G-COLWIDTH` — colonnes ≤ 80 caractères
- `G-FILELEN` — fichiers ~500 lignes
- `G-DATACLASS` — dataclasses dans core/types
- `G-IMMUTABLE` — préférer l'immuabilité
- `G-PROTOCOL` — duck typing via Protocol
- `G-PRIVATE` — refactor en fonctions privées
- `G-TESTS` — tests unitaires isolés
- `G-LOGGING` — logging, pas print
- `G-SECRETS` — secrets via variables d'environnement
- `G-IMPORTS` — imports en couches (lint-imports)
- `G-ONNX` — forward exportable ONNX
- `G-RESOLVEDCONFIG` — config résolue par run
- `G-HYPERPARAMS` — ne pas changer les défauts
- `G-PROFILES` — config-first : tout run via un profil (zéro flag d'hyperparamètre)
- `G-RUNS` — smoke test avant run long
- `G-COMMITS` — commits atomiques
- `G-LOG` — journal des expériences
- `G-ENV` — environnement Poetry sain (mandatory)

---

### G-ENV — environnement Poetry sain (mandatory)
L'environnement se gère **uniquement** par Poetry (`poetry install`,
`poetry run …`) sur Python 3.12/3.13. Le `pyproject.toml` est **PEP 621** :
le champ `[project].requires-python` doit être un spécificateur **PEP 440**
(`>=3.12,<3.14`), **jamais** un caret Poetry (`^3.12`) — ce dernier casse
Poetry 2.x, qui lit désormais `[project]` en priorité (symptôme : `poetry
install`/`poetry lock` échoue au parsing). De même, tout chemin de config
(`[tool.pytest.ini_options].testpaths`, `[tool.coverage]`) doit pointer des
dossiers **existants** (`src`, `test`, `apps` — plus `tools/`, supprimé) sinon
`pytest` casse à la collecte. Après toute édition de `pyproject.toml` :
`poetry lock` puis `poetry install`, et `poetry run pytest` doit collecter
sans erreur de chemin. Ne jamais contourner Poetry par un `pip install`
global.

### G-DOCSTRING — docstrings NumPy
Documenter chaque méthode publique avec une docstring complète au **format
NumPy** (Parameters, Returns, Raises le cas échéant). Une fonction sans
docstring est considérée incomplète.

### G-TYPES — typage strict (Pylance)
Annoter toutes les signatures (paramètres et retours). Le code doit être
**Pylance strict-compliant** : pas de `Any` implicite, pas de retour non
typé. Utiliser `X | None` plutôt que `Optional[X]`.

### G-FORMAT — formatage (black / isort / ruff, PEP 8)
Respecter **PEP 8**. Formater avec **black**, trier les imports avec
**isort**, linter avec **ruff**. Aucune PR ne passe avec un diff de
formatage non appliqué.

### G-MAGIC — pas de magic numbers / strings
Jamais de nombre ou de chaîne « magique » en dur : toujours une **constante
nommée** ou un **enum**. Les constantes de domaine vivent dans
`core/constants`, les enums aussi.

### G-PURE — fonctions explicites, pures, réutilisables
Privilégier des **fonctions explicites**, pures (sans effet de bord caché)
et réutilisables. Une fonction fait une chose ; les dépendances passent en
paramètres plutôt que par état global.

### G-NAMING — pas de noms raccourcis
Pas de raccourcis du type `i`, `m`, `tmp` dans les lambdas, méthodes ou
variables. Les noms décrivent l'intention (`sampleIndex`, pas `i`).

### G-METHODLEN — méthodes ≤ 25 lignes
Aucune méthode ne dépasse **25 lignes**. Au-delà, extraire des fonctions
privées (cf. `G-PRIVATE`).

### G-COLWIDTH — colonnes ≤ 80 caractères
Aucune ligne ne dépasse **80 colonnes**, code comme commentaires.

### G-FILELEN — fichiers ~500 lignes
Viser une **longueur de fichier max ~500 lignes**. Au-delà, scinder par
responsabilité.

### G-DATACLASS — dataclasses dans core/types
Toutes les dataclasses (DTO, configs, types de batch) vivent dans
`ainimator/core/types`. Aucune dataclasse de domaine définie ailleurs.

### G-IMMUTABLE — préférer l'immuabilité
Préférer les structures immuables : `@dataclass(frozen=True)` pour les
valeurs, `NamedTuple` pour les tuples nommés. Muter en place seulement quand
c'est justifié (perf hot-path documentée).

### G-PROTOCOL — duck typing via Protocol
Pour le polymorphisme structurel, utiliser `typing.Protocol` (ex.
`TextEncoderProtocol`) plutôt que l'héritage. L'implémentation se branche par
config, pas par chaîne de classes.

### G-PRIVATE — refactor en fonctions privées
Quand une méthode grossit ou mélange des niveaux d'abstraction, la
**refactorer en fonctions privées** (`_helper`) plutôt que d'allonger le
corps. Garde `G-METHODLEN` satisfait.

### G-TESTS — tests unitaires isolés
Tests **isolés** (pas de dépendances inter-modules), **pytest**, miroir de
l'arbo dans `test/ainimator/`. Catégoriser avec `pytest.mark` (`unit`,
`integration`). Mesurer la couverture :
`pytest --cov=src --cov-report=term-missing`. Tout nouveau code arrive avec
ses tests.

### G-LOGGING — logging, pas print
Utiliser le module `logging`, jamais `print()` dans le code de
bibliothèque. Les CLI configurent le niveau ; le code émet via un logger.

### G-SECRETS — secrets via variables d'environnement
Aucun secret en dur. Lire via `os.environ["KEY"]` (échoue franchement si
absent) ; charger un `.env` en dev si besoin. Ne jamais committer de clé.

### G-IMPORTS — imports en couches (lint-imports)
Respecter l'architecture en couches (ROADMAP §3) : imports **uniquement vers
le bas**, `model` n'importe jamais `data`, `training` est l'unique jonction
data↔model, `cli` sans logique, `legacy/` importable nulle part. Vérifié par
`poetry run lint-imports` (doit rester vert).

### G-ONNX — forward exportable ONNX
Tout `forward()` reste **traçable par `torch.onnx.export`** : pas de control
flow dépendant des données, pas de `.item()` ni de listes Python sur le
chemin du graphe, axes dynamiques déclarés. La boucle autorégressive et le
passage encodeur texte restent **hors** du graphe par frame (un forward = un
pas).

### G-RESOLVEDCONFIG — config résolue par run
Chaque run d'entraînement écrit `resolved_config.yaml` (config complète,
défauts inclus, + git SHA + date) dans son `outputDir`. Aucun run « mystère »
dont on ne peut pas reconstruire la config.

### G-HYPERPARAMS — ne pas changer les défauts
Ne **jamais** modifier les valeurs par défaut d'hyperparamètres existants
sans instruction explicite de Pazimor. Un changement de défaut est une
décision, pas un détail d'implémentation.

### G-RUNS — smoke test avant run long
Lancer le **smoke test overfit-1** (profil `overfit` / `--debug`) avant tout
run long. Ne jamais lancer un run > 30 min sans demande : préparer la
commande (script + config + durée estimée), Pazimor la lance.

### G-COMMITS — commits atomiques
Un commit = un changement cohérent. En cas d'ambiguïté de requirement :
**poser la question**, ne pas choisir silencieusement.

### G-LOG — journal des expériences
Toute expérience conclue ajoute **une ligne** dans `doc/experiments/LOG.md`
(date, run, config, métriques, verdict). Toute décision nouvelle est reportée
dans la fiche concernée (ROADMAP*), avec une date.

### G-PROFILES — config-first : tout run via un profil (zéro flag d'hyperparamètre)
Tout ce qui paramètre un run vit dans un **profil nommé** YAML, **jamais** dans
un flag CLI. Cela vaut pour **toutes** les CLI qui exécutent quelque chose
(entraînement, rollout/génération, `health diagnose`) : la surface CLI se
réduit à `--profile <nom>`, `--config <path>` et le strict minimum
**non-config** par invocation (chemins d'I/O : `--output-dir`, `--checkpoint`,
`--output` ; `--resume`). **Règle dure : pour « juste tester » une valeur,
on crée un nouveau profil, on n'ajoute pas un flag.** Un profil n'est pas
autonome : il part d'une **base** (fusion des YAML canoniques — `network.yaml`,
`dataset.yaml`, etc.) et n'exprime qu'un **delta** (DRY) ; un profil peut
override n'importe quel champ de n'importe quel YAML de la base. Résolution :
`base → delta du profil → resolved_config.yaml` (la trace complète, cf.
`G-RESOLVEDCONFIG`). Aucun `add_argument` d'hyperparamètre (`--embed-dim`,
`--num-clips`, `--scheduled-sampling`, `--frames`, `--control`…) ne doit
réapparaître dans une CLI : si une revue en voit un, c'est un bug à migrer en
champ de profil. Découpage des profils acté (2026-06-24) : `overfit / full /
debug` pour le training ; les profils de génération et de diagnostic suivent
la même mécanique. Cf. `ROADMAP_DETERMINIST.md §3.2`.
