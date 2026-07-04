# AI-nimator — Makefile (recréé en Goal B / phase B5, ROADMAP_PLUGINS §3.4).
#
# Interpréteur : par défaut `poetry run python`. Si l'env poetry est
# cassé sur cette machine, surcharger avec le python du venv :
#   make plugin-unity CHECKPOINT=... \
#     PY=~/Library/Caches/pypoetry/virtualenvs/ai-nimator-*/bin/python

PY ?= poetry run python

.PHONY: plugin-unity plugin-unreal smoke-test test lint-imports monitor

# ---------------------------------------------------------------------
# Goal B — plugins (le CHECKPOINT est OBLIGATOIRE, vérité §2.9)
# ---------------------------------------------------------------------
define REQUIRE_CHECKPOINT
	@test -n "$(CHECKPOINT)" || { \
	  echo "ERREUR: CHECKPOINT requis (aucun défaut)."; \
	  echo "  make $@ CHECKPOINT=output/<run>/checkpoints/<ckpt>.pt"; \
	  exit 2; }
endef

plugin-unity:
	$(REQUIRE_CHECKPOINT)
	$(PY) -m apps.build.build_plugin --target unity \
	  --checkpoint "$(CHECKPOINT)" $(BUILD_FLAGS)

plugin-unreal:
	$(REQUIRE_CHECKPOINT)
	$(PY) -m apps.build.build_plugin --target unreal \
	  --checkpoint "$(CHECKPOINT)" $(BUILD_FLAGS)

# ---------------------------------------------------------------------
# Raccourcis Poetry (réexposés, cf. CLAUDE.md > Commands)
# ---------------------------------------------------------------------
smoke-test:
	$(PY) -m pytest test/ainimator/training/test_controller_training_v2.py -v

test:
	$(PY) -m pytest

lint-imports:
	poetry run lint-imports

monitor:
	poetry run streamlit run apps/monitor/app.py
