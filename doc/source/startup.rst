Comment bien demarrer
=====================

first of all install poetry and init project using:

- ``poetry install``
- ``poetry install --all-extras`` (to run tests and generate the doc)

pipeline de creation
--------------------

1.  pour cree le dataset il faudra en tout premier point telecharger le dataset AMAAS
2.  premier point on uniformise le dataset (voir :doc:`uniformizer`)
        - ``python -m src.cli.uniformizer_cli dir2json raw/ ./converted_60 --target-skel smpl22 --flat --prompts --resample 60``
        - ``python -m src.cli.uniformizer_cli dir2json raw/ ./converted --target-skel smpl22 --flat --prompts --resample 15``
3.  on simplifie l'output 15 fps (voir :doc:`lmf_extractor`)
        - ``python -m src.cli.lmf_extractor_cli bulk ./converted --pattern animation.json --fps-internal 15``
4.  il faut maintenant cree les prompts correspondants au animation deux chois s'offre a vous:

    A.  manuelement: ✍️ (GL HF)
    B.  avec un LLM, j'ai utiliser le playground OpenAI

        - pre :ref:`prompt <prompt-annexe>`.
        - rag_openai_cli.py (avec le dossier output du lmf)

    C.  mais avec un peut de tweak c'est possible de le faire

        - en local (~5j avec une 7900xtx /w ollama cli)
        - avec un autre provider

5.  lancer l'entraînement via :doc:`prompt2anim` (sous-commande ``train``)
        - ⚠️ il faut la VRAM / RAM pour charger le dataset (>48Go) ⚠️ pour ma par j'ai utiliser la plateform AMD Developper Cloud

pipeline de utilisation 
-----------------------

1.  lancer :doc:`prompt2anim` en mode ``sample``
2.  utiliser :doc:`animation_export_cli` pour convertir en BVH/Collada/FBX
