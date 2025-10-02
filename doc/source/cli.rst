Interface CLI
=============

La CLI ``prompt2anim`` fournit les points d'entrée principaux :

.. code-block:: bash

   python -m src.cli.prompt2anim train --help
   python -m src.cli.prompt2anim sample --help

Sous-commandes
--------------

``train``
   Configure et lance l'entraînement via
   ``Prompt2AnimDiffusionTrainer``. Les options incluent les chemins de
   données, la configuration du modèle et la stratégie de contexte.

``sample``
   Charge un checkpoint existant et produit une animation en JSON à
   partir d'un prompt. Cette commande délègue le travail au service
   ``Prompt2AnimDiffusionSampler``.

Chaque sous-commande accepte ``--device`` (``auto`` par défaut) et peut
activer/désactiver CUDA, DirectML ou MPS selon l'environnement.
