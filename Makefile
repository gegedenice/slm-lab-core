# ---- Environment ----
USE_CASE ?= unimarc

# ---- Installation ----
install:
	pip install -e .

# ---- Development ----
prep:
	python -m cli.io build-from-hf $(USE_CASE)

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
