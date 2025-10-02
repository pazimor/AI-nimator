Service d'inférence
===================

``src/features/inferance.py`` fournit le service
``Prompt2AnimDiffusionSampler``. Il assemble :

- la sélection du device (via ``DeviceSelector``) ;
- le chargement du checkpoint et des métadonnées ;
- l'encodage du texte et du tag avec ``PretrainedTextEncoder`` ;
- la génération d'une séquence de rotations 6D grâce à
  ``CausalDiffusion``.

Utilisation minimale
--------------------

.. code-block:: python

   from pathlib import Path
   from src.cli.prompt2anim import buildSamplingConfiguration
   from src.features.inferance import Prompt2AnimDiffusionSampler

   args = ...  # argparse.Namespace
   config = buildSamplingConfiguration(args)
   sampler = Prompt2AnimDiffusionSampler(config)
   sampler.runSampling()

Le résultat est un fichier JSON contenant les quaternions par bone et
par frame, éventuellement enrichi avec les métadonnées (fps, prompt).
