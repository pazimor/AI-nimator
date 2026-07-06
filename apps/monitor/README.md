# AI-nimator Health Monitor

Streamlit dashboard that reads the `health/health.jsonl` stream written in a
run's `outputDir` — by HealthHub for diffusion runs, and by
`ControllerHealthWriter` (`ainimator.health.controller_health_writer`) for
controller runs — and renders verdicts + losses in real time.

## Launch

```bash
# From the repo root (Makefile removed 2026-06-24 — launch directly):
streamlit run apps/monitor/app.py
# Then set the JSONL path in the sidebar
# (e.g. output/<run>/health/health.jsonl).
```

## Architecture

```
apps/monitor/
├── source.py     — RecordSource protocol + JsonlRecordSource / InMemoryRecordSource
├── ingest.py     — JSONL → pandas DataFrame (output_idx X axis)
├── verdicts.py   — verdict aggregation (pure, no Streamlit)
├── losses.py     — multi-loss analysis (pure, no Streamlit)
├── palette.py    — status colors, badges, axis help text
├── panels.py     — Panel protocol + concrete Streamlit panels
├── app.py        — layout + auto-refresh (zero business logic)
└── tests/        — isolated unit tests (no Streamlit)
```

Single allowed import from the main package:
`from ainimator.health.record_schema import …`

## Running tests

```bash
poetry run pytest apps/monitor/tests/ -v
```
