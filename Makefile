# AI-nimator — convenience targets
# Usage: make <target>

.PHONY: smoke-test debug-run lint-imports test

# ---------------------------------------------------------------------------
# smoke-test
#   Run the overfit-1-batch sanity check (§2.7) on a synthetic dataset.
#   No real dataset paths required — runs entirely in pytest's tmp_path.
# ---------------------------------------------------------------------------
smoke-test:
	poetry run pytest \
		test/ainimator/training/test_training_v2.py \
		-k "test_run_overfit_writes_checkpoint or test_checkpoint_round_trips" \
		-v

# ---------------------------------------------------------------------------
# debug-run
#   Fast end-to-end pipeline on a REAL dataset (< 2 min on MPS).
#   Set DATASET_ROOT, TOKENIZER_DIR, OUTPUT_DIR before running.
#   Example:
#     make debug-run \
#       DATASET_ROOT=/Users/pazimor/dataset_preprocessed \
#       TOKENIZER_DIR=output/text/custom_tokenizer \
#       OUTPUT_DIR=output/debug_run
# ---------------------------------------------------------------------------
DATASET_ROOT ?= /path/to/dataset_preprocessed
TOKENIZER_DIR ?= output/text/custom_tokenizer
OUTPUT_DIR ?= output/debug_run

debug-run:
	poetry run python -m ainimator.cli.train_generation_v2 \
		--debug \
		--dataset-root $(DATASET_ROOT) \
		--tokenizer-dir $(TOKENIZER_DIR) \
		--output-dir $(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# lint-imports
# ---------------------------------------------------------------------------
lint-imports:
	poetry run lint-imports

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
test:
	poetry run pytest
