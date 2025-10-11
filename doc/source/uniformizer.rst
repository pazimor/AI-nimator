Script: Uniformizer
===================

Le binaire ``uniformizer`` encapsule les fonctions de conversion et de
retargeting du module ``src.features.uniformizer``. Toutes les commandes
sont accessibles via ``python -m src.cli.uniformizer_cli``.

Usage global
------------

.. code-block:: bash

   python -m src.cli.uniformizer_cli --help

Sous-commandes
--------------

``json2json``
   Retarget un fichier ``animation.json`` respectant la convention rotroot.

   Options principales :

   - ``input`` / ``output`` : chemins source et destination.
   - ``--target-skel smpl22`` : applique le squelette SMPL22.
   - ``--target-map mapping.json`` : mapping personnalisé (chemins joints).
   - ``--resample FPS`` : rééchantillonnage à ``FPS`` images/seconde.
   - ``--prompts`` : génère un ``prompts.json`` adjacent.

``npz2json``
   Convertit un fichier ``.npz`` (positions/quaternions) en rotroot JSON.

   Options identiques à ``json2json`` pour le squelette cible, le
   resampling et les prompts.

``dir2json``
   Parcourt récursivement un dossier contenant des ``.npz`` et produit des
   sous-dossiers ``animation.json``.

   Options supplémentaires :

   - ``--flat`` : écrit chaque export dans un dossier unique ``<stem>_<hash>``.
   - ``--prompts`` : ajoute ``prompts.json`` dans chaque dossier créé.

Cas d'usage
-----------

- Conversion simple ``NPZ ➜ JSON`` :

  .. code-block:: bash

     python -m src.cli.uniformizer_cli npz2json data/clip.npz out/clip.json \
         --target-skel smpl22 --resample 30

- Retarget d'un répertoire complet :

  .. code-block:: bash

     python -m src.cli.uniformizer_cli dir2json dataset/raw dataset/rotroot \
         --target-map config/smpl_map.json --prompts

Diagnostics
-----------

Le CLI remonte les erreurs de ``UniformizerError`` avec un code de sortie
``2``. Les exceptions inattendues retournent ``3`` et affichent le
message ``[UNEXPECTED]``.
