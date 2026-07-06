# Design partagé B7 — encodage de prompt in-engine

> **Statut** : design commun Unity/Unreal de la phase B7
> (`ROADMAP_PLUGINS.md §4`, lancée par Pazimor le 2026-07-04).
> B7 lève l'arbitrage B0 « embeddings pré-calculés seulement » en
> activant l'**extension prévue** par `inference_contract.md §4` :
> un `text_encoder.onnx` séparé dans le bundle (bump **mineur**
> `A7.0 → A7.1` — les gates moteur/orchestrateur, qui filtrent sur le
> majeur `A7`, restent inchangées). Le chemin presets pré-calculés
> (`encode_prompt`) reste valide et prioritaire pour l'authoring ;
> B7 rend le prompt **libre à chaud** invisible pour l'utilisateur :
> `SetPromptText("a person dances")` → tokenize → encode → embedding.

## 1. Le bundle étendu (A7.1)

```
bundle/
├── controller.onnx
├── text_encoder.onnx      # NOUVEAU : input_ids+attention_mask → prompt_emb
├── tokenizer/             # NOUVEAU : fichiers HF verbatim
│   ├── vocab.json         #   49408 entrées BPE
│   └── merges.txt         #   merges classés
├── norm_stats.json        #   (inchangé — prompt.null_emb toujours requis)
├── manifest.json          #   + section optionnelle "text_encoder"
└── presets/
```

- `text_encoder.onnx` : graphe **une passe** — entrées
  `input_ids (B, 32) int64`, `attention_mask (B, 32) float32` ;
  sortie `prompt_emb (B, prompt_emb_channels) float32`. Axe batch
  dynamique, **longueur de séquence figée à `max_length` (32)** — le
  moteur padde toujours à 32, pas d'axe dynamique T. Le graphe inclut
  la tour CLIP figée + la tête de projection + le **mean-pooling
  masqué** (mêmes maths que `encodeTextToPooled` côté training :
  somme(h·mask)/clamp(somme(mask), min=1)). Parité torch↔ORT
  verrouillée à 1e-3 (mesurée 1.3e-7).
- Section manifest (absente ⇒ bundle sans encodeur, comportement A7.0) :

```json
"text_encoder": {
  "file": "text_encoder.onnx",
  "tokenizer": {
    "type": "clip-bpe",
    "vocab": "tokenizer/vocab.json",
    "merges": "tokenizer/merges.txt",
    "max_length": 32,
    "bos_id": 49406,
    "eos_id": 49407,
    "pad_id": 49407
  },
  "pooling": "masked_mean",
  "embedding_channels": 512
}
```

- L'embedding produit alimente le canal prompt existant
  (`SetPromptEmbedding` + cross-fade) : **rien ne change** dans la
  boucle par-frame ni dans la signature du `controller.onnx`.
- Coût embarqué : ~254 Mo fp32 (tour CLIP ViT-B/32). L'encode tourne
  **hors boucle frame** (à l'échelle de l'action), une passe ≈ ms.

## 2. Tokenizer CLIP BPE — algorithme normatif

Source unique : `apps/spec/clip_bpe_reference.py` (référence testée
contre HuggingFace `CLIPTokenizerFast`). Les moteurs la mirrorent
valeur pour valeur ; la parité est verrouillée par
`apps/spec/text_encoding_parity.json` (prompts canoniques → ids),
partagé par pytest, EditMode (Unity) et Automation (Unreal).

### 2.1 Nettoyage et classes de caractères

1. **Lowercase ASCII** (`A-Z → a-z` ; les majuscules non-ASCII passent
   telles quelles — domaine FR/EN, cas marginal documenté).
2. **Collapse des espaces** : toute suite de blancs ASCII
   (` \t\n\r\v\f`) → un espace ; trim aux extrémités.
3. Classes (simplification normative vs `\p{L}` — identique sur le
   domaine FR/EN) :
   - **lettre** = `[a-z]` ou tout codepoint > U+007F ;
   - **chiffre** = `[0-9]` (un token par chiffre) ;
   - **autre** = le reste (ponctuation, runs).

### 2.2 Pré-tokenisation (ordre normatif par position)

contraction (`'s 't 're 've 'm 'll 'd`) > run de lettres > un chiffre
> run de ponctuation. Une contraction interrompt un run de
ponctuation. Les espaces séparent, jamais émis.

### 2.3 Byte-level

Chaque mot → octets UTF-8 → mapping GPT-2 `bytes_to_unicode`
(bijection octet → caractère imprimable). Le **dernier** caractère du
mot reçoit le suffixe `</w>` **avant** les merges (spécificité CLIP
vs GPT-2).

### 2.4 Merges BPE

Fusionner itérativement la paire adjacente de **rang minimal** dans
`merges.txt` (première ligne = rang 0), toutes occurrences gauche →
droite, jusqu'à absence de paire fusionnable. Lookup des pièces dans
`vocab.json`.

### 2.5 Séquence finale

`ids = tronquer(tokens, max_length − 2)` puis
`[bos] + ids + [eos]`, padding avec `pad_id` (= eos) jusqu'à
`max_length` ; `attention_mask` = 1.0 sur bos/ids/eos, 0.0 sur le
padding. Prompt vide → `[bos, eos, pad…]` (valide : équivaut au
prompt vide, PAS au null embedding — pour « aucun prompt », utiliser
`ClearPrompt()` / `prompt.null_emb`).

## 3. Intégration moteur

- Classe pure `ClipBpeTokenizer` (testable sans moteur, chargée depuis
  `tokenizer/` du bundle) + `PromptTextEncoder` (session
  Sentis/NNE sur `text_encoder.onnx`).
- API sur le composant character : `SetPromptText(string) → bool`
  (encodé/échoué). Échec = manifest sans section `text_encoder`
  (bundle A7.0) → log explicite, prompt courant conservé, **jamais**
  de fallback silencieux.
- `SetPromptText` débouche sur le **même** `SetPromptEmbedding`
  (cross-fade 0.3 s, null_emb appris au repos) : les trois chemins
  (preset `prompt_emb`, embedding brut, texte libre) restent
  interchangeables.
- **Distinct de B6** : `SetTextCommand` (mapper bas-niveau → vecteur
  de contrôle) et `SetPromptText` (haut niveau → embedding CLIP)
  coexistent ; aucun des deux ne remplace l'autre.

## 4. Acceptation (roadmap B7)

- bundle `A7.1` exporté avec `--encoder-artifact` : les 3 nouveaux
  fichiers présents, manifest valide (schéma), parité pooled
  torch↔ORT ≤ 1e-3 ;
- mêmes prompts → mêmes ids Unity/Unreal/référence Python
  (`text_encoding_parity.json`, les 12 cas) ;
- `SetPromptText` sur un bundle A7.0 (sans encodeur) échoue
  proprement ; sur un bundle A7.1, produit un embedding fini de la
  bonne dimension et déclenche le cross-fade ;
- chemins presets et `SetPromptEmbedding` inchangés (zéro régression).
