# ROADMAP — Moniteur visuel de santé (`apps/monitor/`, découplé)

> **But** : ajouter au repo un **outil de debug visuel** (page Streamlit locale)
> qui lit le `health/health.jsonl` produit par `health` et le rend lisible :
> verdicts dans le temps + lecture multi-loss.
>
> **Tool entièrement découplé** : il vit **hors du package** (`apps/monitor/`,
> sous `apps/`, sibling de `src/`, `scripts/`, `legacy/`), avec ses propres dépendances et son
> propre lancement. Il ne consomme **que** le contrat JSONL publié par `health`
> et **n'ajoute aucune logique de debug**.
>
> Spec d'implémentation pour un agent de code. Respecte les conventions
> `doc/GUIDELINES.md` (récupérables par ID, ex. `G-DOCSTRING`, `G-TYPES`,
> `G-METHODLEN`, `G-IMPORTS`) **et** les 5 principes S.O.L.I.D. (voir §4).
>
> **Note pivot (2026-06-23, full déterministe)** : le monitor est
> **agnostique au moteur** — il lit le JSONL produit par `health` et découvre
> les axes **dynamiquement** (OCP). Les clés diffusion de l'exemple §2
> (`denoiser_blocks`, `loss_clip_*`, `conditioning_sensitivity`) sont
> **illustratives/historiques** : le moteur déterministe émet à la place
> `controller_blocks`, `control_sensitivity`, `mean_collapse`,
> `rollout_drift` et les losses contrôleur (L2 vitesses / géodésique 6D /
> velocity / foot-contact). Aucune ligne du monitor n'est à changer pour
> autant — c'est précisément le but du découplage par contrat.

---

## Alignement avec le ROADMAP projet (pas de changement de direction)

Vérifié contre `doc/ROADMAP.md` (§4 — surcouche santé) :

- Le ROADMAP prévoit déjà `ainimator.cli.health watch` = « suivre un run en
  cours via ses JSONL ». **Le monitor est le front-end visuel de ce `watch`**,
  pas un outil concurrent.
- Décision canonique §4 : *« `health/` est l'outil de debug global et **unique**…
  un seul point d'entrée CLI »*. Le monitor **respecte** cette règle car il
  **ne contient aucune logique de debug** : probes, contracts et **verdicts
  restent calculés par `health`**. Le monitor est un pur *rendu* de leur sortie.
- Conséquence : `health` reste **propriétaire** du contrat et des verdicts.
  Le monitor n'est qu'un consommateur en lecture seule. Le découplage total
  **renforce** cette séparation au lieu de la diluer.

---

## 0. Le principe directeur : un contrat commun (DIP)

`health` **écrit** le JSONL ; le monitor le **lit**. Aujourd'hui le schéma est
**implicite** : le préfixe `verdict.` est codé en dur dans `hub.py:362`
(`record[f"verdict.{res.name}"] = ...`), les préfixes `loss_` / `loss_share.`
le sont ailleurs. Si le monitor ré-encode ces littéraux de son côté, les deux
dérivent → bug silencieux le jour où un préfixe change.

**Décision SOLID (Dependency Inversion)** : extraire le schéma JSONL dans une
**abstraction partagée** dont *dépendent les deux* — `health` (producteur) et
`monitor` (consommateur) — au lieu de se dépendre l'un l'autre ou de dupliquer
des chaînes. C'est ça, le « contrat commun ».

```
            ┌──────────────────────────┐
            │   schéma JSONL (contrat)  │   ← abstraction stable, sans torch
            │  préfixes + Verdict enum  │      publié par health.record_schema
            └────────────▲───────▲──────┘
        dépend de        │       │        dépend de (1 seul import)
        ┌────────────────┘       └────────────────┐
   ainimator.health (écrit)      apps/monitor (lit, hors package)
```

Le monitor est découplé au niveau **process / packaging / cycle de vie** ; son
**unique** point de couplage est cette abstraction publiée — ce qui est
exactement le but de DIP, pas une entorse. Aucune dépendance vers `hub.py`,
`probe.py` ou le reste de `health`.

---

## 1. Où ça vit dans le repo

### 1.1 Le contrat partagé (dans `health`, pas dans le monitor)
Nouveau module **`ainimator/health/record_schema.py`** (couche L4, à côté de
`contract.py`). Pur Python, sans torch. C'est la **surface publiée** que `health`
expose et que le monitor importe. Contient :

- les **constantes de préfixe** : `STEP_KEY = "step"`, `VERDICT_PREFIX =
  "verdict."`, `LOSS_PREFIX = "loss_"`, `LOSS_SHARE_PREFIX = "loss_share."`,
  `EVERY_STEPS_DEFAULT = 50` ;
- le **re-export de `Verdict`** (déjà dans `contract.py`) ;
- des **helpers purs** de lecture : `verdict_axis(key) -> str | None`,
  `is_active_loss(values) -> bool`, `group_record(raw) -> ParsedRecord`.

> **Petit refactor du hub (validé)** : `hub.py` (ligne 362) et `jsonl_writer.py`
> doivent **utiliser** `VERDICT_PREFIX` au lieu du littéral `"verdict."`.
> Objectif : *exposer* le contrat, pas changer le comportement. **Sortie JSON
> strictement identique** (test de non-régression byte-à-byte sur une ligne).
> Cohérent avec le principe : `health` reste propriétaire du contrat. Commit atomique séparé.

### 1.2 Le tool monitor — HORS package
Nouveau dossier **`apps/monitor/`** (sous `apps/`, sibling de `src/`,
`scripts/`, `legacy/`), **hors du package `ainimator`**. Conséquences :

- `import-linter` ne le gouverne pas (ce n'est pas `ainimator.*`) → aucun ajout
  de couche, aucune modif des contrats existants. Le périmètre du package reste intact.
- Son **unique** dépendance vers le repo : `from ainimator.health.record_schema
  import …` (le contrat). Interdiction d'importer quoi que ce soit d'autre de
  `ainimator` (vérifié par un test du monitor, voir §10).
- Lancement autonome : `poetry run streamlit run apps/monitor/app.py`.
  Aucune entrée ajoutée à `ainimator.cli`. *(Le Makefile de confort a été
  supprimé le 2026-06-24 ; une cible `make monitor` reviendra avec
  l'orchestrateur de build `apps/build/`, Goal B / B5.)*

```
src/ainimator/health/record_schema.py   (NOUVEAU — contrat publié, dans le package)
apps/monitor/                          (NOUVEAU — tool découplé, hors package)
├── app.py  ingest.py  verdicts.py  losses.py  panels.py  palette.py
├── tests/  requirements.txt  README.md
```

### 1.3 Dépendances (même env Poetry)
**Décision (Pazimor)** : Streamlit / plotly sont des **dépendances Poetry
principales** du projet (installées avec tout, un seul `poetry install`). Le
découplage reste **architectural** (tool hors package, un seul import de contrat),
pas environnemental — ce qui est suffisant et plus simple. Pas de venv ni de
`requirements.txt` séparés. Le cœur testable (`ingest`/`verdicts`/`losses`) reste
néanmoins **sans** Streamlit (testable en isolation, principe SRP).

---

## 2. Le contrat de données (schéma réel, relevé sur `health.jsonl`)

Clés **à plat**, nommées `groupe.sous-clé`. Une ligne = un output émis
**toutes les 50 steps**.

```jsonc
{
  "step": 50,                          // ⚠ compteur INTRA-epoch (alterne 50/100), PAS global
  "denoiser_blocks.mean": 0.439, "denoiser_blocks.std": 22.85,
  "denoiser_blocks.norm": 26163.07, "denoiser_blocks.effective_rank": 0.0161,
  "denoiser_blocks.intra_batch_sim": 0.961,
  "output_head.mean": -0.0057, "output_head.std": 0.321,
  "output_head.norm": 168.97, "output_head.update_ratio": 0.198,

  "loss_total": 6.048,
  "loss_bone": 0.403, "loss_global": 0.231,
  "loss_clip_guidance": 4.551, "loss_clip_aux_pool": 1.967,
  "loss_vel_xyz": 0.0, "loss_joint_xyz": 0.0,            // 0.0 = INACTIVES (masquées)
  "loss_foot_contact": 0.0, "loss_x0_contrastive": 0.0,

  "loss_share.bone": 0.066, "loss_share.global": 0.038,  // parts DÉJÀ calculées
  "loss_share.clip_guidance": 0.732, "loss_share.clip_aux_pool": 0.162,
  "loss_share": 0.038,

  "conditioning_sensitivity": 0.00017, "intra_batch_sim": 0.961,
  "effective_rank": 0.0161, "update_ratio": 0.198,

  "verdict.conditioning_sensitivity": "CRITICAL",        // verdicts FOURNIS par health
  "verdict.update_ratio": "WARNING",
  "verdict.loss_decomposition": "OK"
  // verdict.cond_uncond_sim / verdict.post_norm_stats : renseignés HORS entraînement
  //   → toujours UNKNOWN ici → IGNORÉS par défaut (filtrer les UNKNOWN constants)
}
```

**Faits structurants**

- 🔴 **Verdicts fournis, pas recalculés.** `verdict.*` ∈ `{OK, WARNING, CRITICAL, UNKNOWN}`. Découverte **dynamique** des axes (OCP) ; un axe constamment `UNKNOWN` est masqué.
- 🔴 **`step` non global** → axe X = `output_idx` (ordre des lignes). Option `global_step ≈ output_idx × 50`.
- Composantes de loss toujours nulles = **inactives** → grisées/masquées.
- Mapping verdict → métrique (drill-down §5) : `conditioning_sensitivity`→`conditioning_sensitivity` ; `update_ratio`→`update_ratio`,`output_head.update_ratio` ; `loss_decomposition`→`loss_share.*`.

---

## 3. Architecture du module (responsabilités séparées — SRP)

```
apps/monitor/                  # HORS package ainimator
├── source.py            # RecordSource (protocol) + JsonlRecordSource
├── ingest.py            # parse tolérant + group_record → DataFrame + output_idx
├── verdicts.py          # agrégation des verdicts FOURNIS (pur, testé)
├── losses.py            # analyse multi-loss : dominante / limitante (pur, testé)
├── panels.py            # Panel (protocol) + panneaux concrets (rendu Streamlit)
├── app.py               # composition : layout + auto-refresh (zéro logique métier)
├── palette.py           # couleurs de statut, palette losses, AXIS_HELP
├── tests/               # test_ingest / test_verdicts / test_losses / test_decoupling
└── README.md            # `streamlit run apps/monitor/app.py`
```

Pas de `requirements.txt` : Streamlit/plotly vivent dans le `pyproject.toml`
du projet (deps Poetry principales).

Seul import autorisé vers le repo : `ainimator.health.record_schema`
(le contrat). `group_record` vient de là — pas de réimplémentation locale.

- **`source.py` / `ingest.py`** : *acquisition*. Une seule raison de changer = le format/source des données.
- **`verdicts.py`** : *agrégation des verdicts fournis* (jamais de calcul de seuil).
- **`losses.py`** : *analyse multi-loss* (dominante via `loss_share`, limitante via part × pente).
- **`panels.py` / `app.py`** : *présentation*. Séparées du calcul → testables indépendamment.

---

## 4. Mapping S.O.L.I.D. → décisions concrètes

**S — Single Responsibility.** Un fichier = une raison de changer : acquisition
(`ingest`), agrégation verdicts (`verdicts`), analyse loss (`losses`), rendu
(`panels`), composition (`app`). Aucune fonction > 25 lignes (`G-METHODLEN`).

**O — Open/Closed.** Les axes de verdict et les losses sont **découverts
dynamiquement** depuis le record : ajouter un axe dans `health` n'exige **aucune
modif** du monitor. Les panneaux sont enregistrés via une liste de `Panel` →
ajouter une vue = ajouter une classe, sans toucher `app.py`.

**L — Liskov.** `RecordSource` (protocol) a deux implémentations substituables :
`JsonlRecordSource` (prod) et `InMemoryRecordSource` (tests, samples). Tout
`Panel` respecte la même signature `render(ctx)` → interchangeables.

**I — Interface Segregation.** Protocoles **étroits** : `RecordSource.read()`,
`VerdictView`, `LossView`. L'UI dépend du strict nécessaire, pas d'un god-object
`HealthHub`. Le monitor n'importe **que** `record_schema`, rien d'autre de `health`
(un test de découplage l'assure, §10).

**D — Dependency Inversion.** Producteur (`health`) et consommateur
(`apps/monitor`) dépendent tous deux de l'abstraction `record_schema` (préfixes
+ `Verdict`), jamais l'un de l'autre. Le monitor dépend du **contrat publié**,
pas des internes de `hub.py` — et vit hors package pour rendre ce couplage
minimal explicite.

---

## 5. Verdicts : afficher & tracer (besoin #2)

Couleurs : `CRITICAL` 🔴 / `WARNING` 🟠 / `OK` 🟢 / `UNKNOWN` ⚪ (ignoré pour le « pire »).

- **5.1 État courant** : une pastille par axe (dernière ligne) + **badge global** = pire axe hors `UNKNOWN`.
- **5.2 Heatmap temporelle** (la vue debug) : X = `output_idx`, Y = axe, couleur = statut → on voit *quand* un axe passe CRITICAL→OK et si des axes se dégradent ensemble.
- **5.3 Drill-down** : sélection d'un axe → métrique(s) sous-jacente(s) + bandes de statut.

`verdicts.py` (pur, sans Streamlit) : `current_verdicts(df)`, `worst_now(df)`,
`verdict_matrix(df)`, `transitions(df)`. Pas de seuils ici — ils vivent dans `health`.

---

## 6. Multi-loss : qui est limitant (besoin #3)

Composantes actives : `loss_bone`, `loss_global`, `loss_clip_guidance`,
`loss_clip_aux_pool` (les `*_xyz`, `foot_contact`, `x0_contrastive` sont à 0.0 → masquées).

- **Superposition** (X = `output_idx`), options log + normalisation 0-1.
- **Contribution** : aire empilée directe depuis `loss_share.*` (déjà fourni).
- **Limitante** : badge 🔒 sur la loss à forte part `loss_share` **et** pente récente ≈ 0/positive pendant que `loss_total` ralentit.
- **Table** triée par part : loss | valeur | part | Δ récent | tendance | active/inactive.

`losses.py` (pur, testé) : `active_losses(df)`, `contribution(df)`,
`limiting_loss(df)`.

---

## 7. UI Streamlit (`app.py` + `panels.py`)

Layout : header (output courant + badge global) → pastilles de verdict →
**heatmap verdicts** → multi-loss (superposition + contribution + 🔒) → KPIs
(`loss_total`, `update_ratio`, `effective_rank`, `conditioning_sensitivity`) →
drill-down → table brute (repliable).

UX : auto-refresh 2–5 s (toggle + bouton) ; **X = output_idx** ; sélecteur de
fenêtre (N derniers / dernière epoch / tout) ; chemin JSONL en sidebar (défaut =
`output/<run>/health/health.jsonl`) ; thème sombre, couleurs de statut fixes.

---

## 8. Phases

> **État (2026-07-01) : P0–P3 livrés, P4 partiel.** `health/record_schema.py`
> existe et `apps/monitor/` est complet (source / ingest / verdicts / losses /
> palette / panels / app + tests isolés) ; `streamlit`/`plotly` sont dans
> `pyproject.toml` ; le sélecteur de fenêtre est en place. Restes P4 :
> lecture incrémentale (offset), persistance du chemin, annotations de
> transitions de verdict. **Lacune fermée le 2026-07-01 :** la voie
> contrôleur n'écrivait aucun `health.jsonl` (le moniteur ne voyait que la
> diffusion) — corrigé par `ControllerHealthWriter`
> (`health/controller_health_writer.py`), branché dans les deux boucles
> contrôleur. Le lancement se fait directement
> (`streamlit run apps/monitor/app.py`) — la cible `make monitor` reste
> reportée au Makefile refait (Goal B / B5).

**P0 — Contrat partagé (dans `health`).** Créer `health/record_schema.py`
(constantes + helpers + re-export `Verdict`). **Petit refactor** : `hub.py:362`
et `jsonl_writer.py` consomment `VERDICT_PREFIX` au lieu du littéral — **sortie
JSON byte-à-byte identique** (test de non-régression avant/après). `lint-imports`
reste vert (aucune nouvelle couche). *Commit atomique, dans le package.*

**P1 — MVP « ça vit » (`apps/monitor/`).** `source.py` + `ingest.py` (via
`group_record` du contrat) + `app.py` minimal (courbe `loss_total`,
X = `output_idx`, auto-refresh). ✅ avance en live, X correct.

**P2 — Verdicts (besoin #2).** `verdicts.py` + pastilles + heatmap + drill-down.
Tests sur 2 samples (tranche « début » CRITICAL, tranche « fin » OK). ✅ heatmap
montre la transition.

**P3 — Multi-loss (besoin #3).** `losses.py` + superposition + contribution +
🔒 limitante + masquage inactives. ✅ on identifie la dominante/limitante.

**P4 — Finition.** Lecture incrémentale (offset), sélecteur de fenêtre, thème,
persistance du chemin, annotation des transitions de verdict sur les courbes,
ajout de `streamlit`/`plotly` au `pyproject.toml` (cible `make monitor`
reportée au Makefile refait avec `apps/build/`, Goal B / B5).

---

## 9. Conventions à respecter

**Côté `health/record_schema.py` (dans le package → `doc/GUIDELINES.md` s'applique)**
- Docstrings **NumPy**, **Pylance strict**, **pas de magic strings** (les préfixes EN deviennent les constantes), méthodes ≤ 25 lignes, ≤ 80 colonnes.
- `poetry run lint-imports` **vert** ; non-régression de la sortie JSON ; commit atomique.

**Côté `apps/monitor/` (hors package)**
- Même style (docstrings NumPy, types stricts, ≤ 25 lignes, ≤ 80 col, ≈ ≤ 500 lignes/fichier) pour rester cohérent avec le repo.
- **Tests isolés** dans `apps/monitor/tests/`, sans Streamlit ; samples = tranches du vrai `health.jsonl`.
- Décision ambiguë → demander ; ne jamais changer un défaut existant de `health`.

---

## 10. Critères d'acceptation

- [ ] `record_schema.py` est l'**unique** source des préfixes ; `hub.py` l'utilise ; **JSON de sortie inchangé** (test de non-régression byte-à-byte).
- [ ] `lint-imports` reste vert **sans modification des contrats** (le monitor est hors package).
- [ ] **Test de découplage** : le monitor n'importe de `ainimator` **que** `health.record_schema` (vérifié par `apps/monitor/tests/test_decoupling.py`).
- [ ] Lance via `poetry run streamlit run apps/monitor/app.py`, lit le vrai `health.jsonl` sans crash (ligne tronquée / `step` non global / NaN).
- [ ] Verdicts **affichés et tracés** (pastilles + heatmap + drill-down), axes découverts dynamiquement, `UNKNOWN` constants masqués. **Aucun recalcul.**
- [ ] Multi-loss : contribution via `loss_share`, badge 🔒 limitante, inactives masquées.
- [ ] `ingest`/`verdicts`/`losses` **testés sans Streamlit** ; `streamlit`/`plotly` ajoutés au `pyproject.toml` ; aucune écriture dans le JSONL.

---

## 11. Décisions (toutes tranchées)

| # | Décision | Statut |
|---|---|---|
| 1 | **Tool hors package** dans `apps/monitor/` (découplage architectural). | ✅ validé |
| 2 | **Petit refactor `hub.py`** pour exposer le contrat via `VERDICT_PREFIX`, sortie JSON identique, commit séparé. | ✅ validé |
| 3 | **`Verdict` reste dans `health/contract.py`**, re-exporté depuis `record_schema` (pas de descente en `core`). | ✅ validé |
| 4 | **Deps Poetry principales** (`streamlit`/`plotly` dans `pyproject.toml`, installées avec tout). Pas de venv séparé. | ✅ validé |

→ Plus de décision bloquante : la roadmap est prête pour l'implémentation (Sonnet).
