Script: Animation Export
========================

``animation_export_cli`` convertit un fichier ``animation.json``
(rotroot) vers plusieurs formats d'échange standards (FBX/frame binary,
BVH, Collada/DAE). L'entrée point se fait via :

.. code-block:: bash

   python -m src.cli.animation_export_cli <input.json> <output.ext> [--format]

Fonctionnement
--------------

Le format est déduit de l'extension ``output.ext``. Les extensions
supportées et leurs alias sont :

- ``.fb`` (alias ``fbx``) : archive binaire compressée contenant positions
  et rotations.
- ``.bvh`` : format texte BVH pour l'import dans les DCC.
- ``.dae`` (alias ``collada``) : fichier Collada 1.4.1.

Options principales
-------------------

- ``input`` : chemin du fichier ``animation.json`` rotroot (positions,
  quaternions, bone names, FPS).
- ``output`` : chemin de destination (extension utilisée pour inférer la
  cible).
- ``--format`` : force le format lorsqu'on souhaite un suffixe différent
  (``fbx``, ``fb``, ``bvh``, ``collada``, ``dae``).

Exemples
--------

- Export FBX/frame binary :

  .. code-block:: bash

     python -m src.cli.animation_export_cli clip.json clip.fb

- Export Collada en forçant le format :

  .. code-block:: bash

     python -m src.cli.animation_export_cli clip.json exports/clip_mesh \
         --format collada

- Chaîne complète après uniformisation :

  .. code-block:: bash

     python -m src.cli.uniformizer_cli npz2json raw/clip.npz clip.json
     python -m src.cli.animation_export_cli clip.json clip.bvh

Codes de sortie
---------------

- ``0`` : export réussi (message ``[export] format -> chemin``).
- ``2`` : erreur fonctionnelle (ex : format non supporté, JSON invalide).
- ``3`` : exception inattendue avec le message ``[unexpected]``.
