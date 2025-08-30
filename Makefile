.PHONY: venv install prep train eval smoke clean

# ---- Environment ----
USE_CASE ?= unimarc
VENV := .venv

# ---- Load .env file ----
ifneq (,$(wildcard .env))
include .env
export
endif

# ---- Installation ----
$(VENV):
	uv venv $(VENV)
	uv pip install -U pip wheel

venv: $(VENV)

install: $(VENV)
	uv sync

# ---- Development ----
prep: install
	uv run python -m cli.io build-from-hf $(USE_CASE)

train: install
	uv run python -m cli.finetune run $(USE_CASE)

eval: install
	uv run python -m cli.evaluate run $(USE_CASE)

run-gradio: install
	uv run gradio app.py

train-hf-job: install
	uv run python scripts/run_hf_job.py $(USE_CASE)

# ---- Testing ----
test: install
	uv run pytest -q

smoke: install
	uv run bash use_cases/$(USE_CASE)/scripts/smoke_eval.sh

# ---- Housekeeping ----
clean:
	rm -rf $(VENV) .venv.lock *.lock
