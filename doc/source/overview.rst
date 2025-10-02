Vue d'ensemble
==============

``AI-nimator`` est un outil de génération d'animations conditionnées par
texte. Il repose sur un pipeline Python 3.12, PyTorch et NumPy, avec une
architecture orientée services :

- ``src/shared`` regroupe les utilitaires réutilisables (sélection de
  device, conversions quaternioniques, text encoder, diffusion temporelle).
- ``src/features`` fournit les services dédiés :
  - ``training.py`` orchestre la préparation des données, le caching et la
    boucle d'entraînement du modèle de diffusion.
  - ``inferance.py`` expose les services d'inférence basés sur le même
    backbone.
- ``src/cli`` contient la CLI ``prompt2anim`` qui connecte les
  configurations utilisateur aux services.

Les sections suivantes détaillent l'utilisation de chaque composant.
