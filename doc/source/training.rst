Service d'entraînement
======================

Le module ``src/features/training.py`` expose le service
``Prompt2AnimDiffusionTrainer`` chargé de :

1. Initialiser les composants partagés (encodeur texte, dataset,
   backbone de diffusion).
2. Pré-cacher les données à l'aide de ``DatasetCacheBuilder`` pour
   optimiser le débit d'entraînement.
3. Gérer la boucle d'entraînement, l'évaluation périodique et la
   sérialisation des checkpoints via ``CheckpointManager``.

Configuration
-------------

Les paramètres sont regroupés dans ``TrainingConfiguration``
(``src/shared/types``) et couvrent :

- chemins (données, sauvegardes) ;
- hyperparamètres modèle / diffusion ;
- options de contexte et de validation ;
- paramètres device via ``DeviceSelectionOptions``.

La méthode ``runTraining`` orchestre l'ensemble du cycle :

.. code-block:: python

   from src.cli.prompt2anim import buildTrainingConfiguration
   from src.features.training import Prompt2AnimDiffusionTrainer

   args = ...  # argparse.Namespace
   config = buildTrainingConfiguration(args)
   trainer = Prompt2AnimDiffusionTrainer(config)
   trainer.runTraining()
