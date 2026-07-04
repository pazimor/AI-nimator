# Design partagé B6 — texte libre → vecteur de contrôle

> **Statut** : design commun Unity/Unreal de la phase B6
> (`ROADMAP_PLUGINS.md §4`, lancée par Pazimor le 2026-07-04).
> ⚠ B6 ne concerne **pas** le canal prompt (embedding, haut niveau —
> `rig_binding.md §3`) : elle mappe une phrase sur le **vecteur de
> contrôle bas-niveau** (vx, vz[, aim]). Le chemin presets reste le
> défaut ; ce mapper est un raccourci d'ergonomie, jamais un passage
> obligé.

## 1. Source unique : la table canonique

`apps/spec/text_to_control.json` est la **seule** définition des
mots-clés (directions FR/EN, vitesses FR/EN, défauts). Chaque plugin
embarque une **copie verbatim** (asset texte) avec un commentaire
pointant ici ; toute évolution passe par ce fichier puis se propage.
L'implémentation de référence (testée) est
`apps/spec/text_to_control_reference.py` — les moteurs la mirrorent
valeur pour valeur.

## 2. Algorithme (déterministe, ordre normatif)

1. **Normaliser** : minuscules ; retirer les accents (é→e, è→e, à→a,
   ç→c…) ; tokeniser sur tout caractère non alphabétique.
2. **Directions** : pour chaque token présent dans `directions`,
   accumuler le vecteur (somme). Somme non nulle → normaliser à
   l'unité (les diagonales « avance à gauche » donnent (−0.707,
   0.707)). Somme nulle alors que des mots de direction étaient
   présents (« gauche droite ») → **ambigu** = résolution échouée.
3. **Vitesse** : parmi les tokens présents dans `speeds`, prendre la
   valeur **maximale** (« cours lentement » → 0.1 ; règle simple et
   déterministe). `stop`-family = 0.0 et **gagne toujours** (si un
   token de vitesse 0 est présent, la vitesse est 0).
4. **Défauts** : direction sans vitesse → `speed_when_direction_only`
   (0.033). Vitesse sans direction → `direction_when_speed_only`
   (avant). Vitesse 0 → contrôle (0, 0), direction ignorée.
5. **Sortie** : `control = direction × vitesse` en (vx, vz) **bruts**
   (m/frame — le moteur z-normalise comme pour un preset). Si le
   bundle a 4 canaux, `aim = direction` (ou (0,1) si vitesse 0).
6. **Aucun token reconnu** (ni direction ni vitesse), ou ambiguïté
   (règle 2) → **résolution échouée** : le mapper retourne « pas de
   résultat », le contrôle courant est conservé, un log l'explique.
   Jamais de fallback silencieux vers un mouvement.

## 3. Intégration moteur

- Classe pure `TextToControlResolver` (testable sans moteur), chargée
  depuis la copie embarquée du JSON.
- API sur le composant character : `SetTextCommand(string) → bool`
  (résolu/échoué). Champ texte optionnel dans
  l'Inspector/Details pour tester en Play/PIE.
- **Zéro impact preset** : `SetTextCommand` écrit le même état de
  contrôle que `SetPreset` ; les deux chemins restent
  interchangeables frame par frame.

## 4. Hors scope (v1)

- Résolution par embedding (phrase → preset le plus proche via CLIP) :
  possible côté authoring Python plus tard, pas dans le moteur.
- Le prompt haut-niveau : déjà couvert (`rig_binding.md §3`,
  `encode_prompt`).

## 5. Acceptation (roadmap B6)

- « cours vers la gauche » → contrôle plausible ((−0.1, 0) ici) ;
- déterministe et documenté (ce fichier + la table) ;
- chemin presets intact ;
- parité : mêmes phrases → mêmes vecteurs Unity/Unreal/référence
  Python (les tests des trois côtés partagent les mêmes cas).
