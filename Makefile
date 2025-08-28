# ---- Environment ----
USE_CASE ?= unimarc

# ---- Installation ----
install:
	pip install -e .

# ---- Development ----
prep:
	cd use_cases/$(USE_CASE) && \
	PYTHONPATH=. python -m slm_lab_unimarc.io.loader_hf \
		--repo Geraldine/metadata-to-unimarc-reasoning \
		--train data/processed/train.jsonl \
		--eval data/eval/heldout.jsonl \
		--eval-ratio 0.2

train:
	python -m cli.finetune run $(USE_CASE)

eval:
	# TODO: Adapt cli/evaluate.py to use use-cases
	echo "Not implemented yet"

run-gradio:
	gradio app.py

# ---- Testing ----
test:
	pytest -q
