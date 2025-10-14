Script: LMF Extractor
=====================

``lmf_extractor`` génère des descripteurs *LMF* (Language Motion
Features) à partir des exports ``animation.json`` produits par
``uniformizer``. L'entrée point se fait via ``python -m
src.cli.lmf_extractor_cli``.

Extraction simple
-----------------

.. code-block:: bash

   python -m src.cli.lmf_extractor_cli path/to/animation.json \
       --fps-internal 15.0

Arguments :

- ``input`` : chemin du fichier ``animation.json``.
- ``-o / --output`` : fichier ``.lmf.json`` de sortie (par défaut, même
  dossier que l'entrée).
- ``--fps-internal`` : cadence de travail interne pour l'analyse (15 FPS
  par défaut).

Mode bulk
---------

Pour traiter un répertoire complet, utilisez la sous-commande ``bulk`` :

.. code-block:: bash

   python -m src.cli.lmf_extractor_cli bulk dataset/rotroot \
       --pattern animation.json --fps-internal 20 \
       --follow-links --no-skip-up-to-date

Options ``bulk`` :

- ``root`` : dossier racine à parcourir (``.`` par défaut).
- ``--pattern`` : motif de fichier (insensible à la casse), ``animation.json``
  par défaut.
- ``--fps-internal`` : cadence de traitement.
- ``--follow-links`` : suit les liens symboliques.
- ``--skip-up-to-date`` / ``--no-skip-up-to-date`` : contrôle la
  régénération des fichiers déjà à jour (activé par défaut).

Codes de retour
---------------

- ``0`` : succès.
- ``1`` : erreur lors de l'extraction (détails sur la sortie d'erreur).

Lorsque ``bulk`` est utilisé, ``run_bulk`` lève des exceptions Python en
cas d'échec (chemins inaccessibles, absence d'écriture…). Assurez-vous
de disposer des droits en lecture/écriture suffisants.
