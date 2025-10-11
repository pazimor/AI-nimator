Script: Prompt to Anim
======================

La commande ``prompt2anim`` regroupe les opérations d'entraînement et
d'inférence du modèle de diffusion texte → animation. Elle se lance via :

.. code-block:: bash

   python -m src.cli.prompt_to_anim_cli <sous-commande> [options]

Sous-commandes
--------------

``train``
   Prépare le dataset, instancie les composants (encodeur texte,
   diffusion, cache) puis lance la boucle d'entraînement.

   Paramètres essentiels :

   - ``--data-dir`` *(obligatoire)* : dossier contenant les paires
     ``animation.json`` / ``prompts.json``.
   - ``--save-dir`` (``./ckpts`` par défaut) : répertoire des checkpoints.
   - ``--epochs``, ``--batch-size``, ``--lr`` : hyperparamètres d'apprentissage.
   - ``--seq-frames`` : longueur des segments utilisés pour l'entraînement.
   - ``--context-last`` / ``--context-train-mode`` / ``--context-train-ratio`` :
     configurent l'utilisation du contexte temporel.
   - ``--d-model`` ``--layers`` ``--moe-experts`` ``--moe-topk`` : taille du
     réseau et paramètres Mixture-of-Experts.
   - ``--text-model`` : nom Hugging Face de l'encodeur texte utilisé.
   - ``--val-split`` ``--val-every`` : fraction et fréquence des évaluations.
   - ``--success-deg`` ``--target-success`` : seuil et objectif pour le taux
     de réussite (arrêt anticipé possible).
   - Options device: ``--device`` (``auto``), ``--no-cuda``, ``--no-dml``,
     ``--no-mps``, ``--strict-device``.

   Exemple :

   .. code-block:: bash

      python -m src.cli.prompt_to_anim_cli train \
          --data-dir data/rotroot --save-dir runs/ckpts --exp-name exp01 \
          --epochs 10 --batch-size 4 --seq-frames 240 --context-last 2

``sample``
   Charge un checkpoint et génère un fichier ``animation.json`` à partir
   d'un prompt texte.

   Options principales :

   - ``--ckpt`` *(obligatoire)* : chemin vers le checkpoint ``.pt``.
   - ``--prompts`` *(obligatoire)* : fichier contenant un ou plusieurs prompts.
   - ``--out-json`` : fichier de sortie rotroot.
   - ``--frames`` : nombre d'images générées.
   - ``--steps`` : nombre d'étapes de diffusion.
   - ``--guidance`` : échelle de guidance classifier-free.
   - ``--context-jsons`` : liste de ``animation.json`` fournis en contexte
     (séparés par des virgules).
   - ``--bones`` : ordre des articulations à utiliser lorsqu'aucun contexte
     n'est fourni.
   - ``--omit-meta`` : supprime la section ``meta`` du JSON généré.
   - ``--text-model`` : surcharge l'encodeur texte du checkpoint.

   Exemple :

   .. code-block:: bash

      python -m src.cli.prompt_to_anim_cli sample \
          --ckpt runs/ckpts/exp01_best.pt --prompts prompts.json \
          --out-json outputs/clip.json --frames 300 --steps 16

Codes de sortie
---------------

- ``0`` : exécution réussie (entraînement ou sampling).
- ``1`` : erreur inattendue côté Python.
- Les vérifications internes lèvent des exceptions avec messages détaillés
  (device indisponible, dataset invalide, etc.).
