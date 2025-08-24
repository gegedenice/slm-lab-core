# slm-lab-core

Boîte à outils générique pour fine-tuning de Small Language Models (SLM).

## Fonctionnalités
- Préparation de données (templates instruction & chat)
- Fine-tuning LoRA (CPU/GPU)
- Distillation (teacher → student)
- Évaluation (rouge, exact match, BERTScore, XML-validité…)
- Packaging & service API

## Installation
```bash
pip install -e .
```

## Arborescence

```
slm-lab-core/
├─ README.md
├─ pyproject.toml
├─ setup.cfg          # style/lint optionnel
├─ Makefile
├─ .gitignore
├─ src/
│  └─ slmlab/
│     ├─ __init__.py
│     ├─ utils/
│     │  └─ config.py
│     ├─ prep/
│     │  ├─ __init__.py
│     │  └─ templating.py      # registre générique de templates
│     ├─ train/
│     │  ├─ __init__.py
│     │  ├─ sft_lora.py        # fine-tune LoRA
│     │  ├─ distill.py         # distillation (teacher->student)
│     │  ├─ cpt.py             # continual pretraining (option)
│     │  └─ teacher.py         # interface teacher
│     ├─ eval/
│     │  ├─ __init__.py
│     │  ├─ metrics.py         # exact_match, rouge, bertscore
│     │  ├─ xml_eval.py        # validité XML / coverage
│     │  └─ runner.py          # eval baseline vs tuned
│     ├─ postproc/
│     │  └─ __init__.py        # ex: serializeurs JSON→XML
│     └─ serve/
│        └─ fastapi_app.py     # API générique de génération
└─ src/cli/
   ├─ __init__.py
   ├─ finetune.py              # CLI fine-tuning
   ├─ distill.py               # CLI distillation
   ├─ prep.py                  # CLI data prep générique
   ├─ evaluate.py              # CLI évaluation
   └─ demo.py                  # petite API/streamlit

```

## Utilisation rapide

```
# Fine-tuning
python -m slmlab.cli.finetune run --cfg_path configs/default.yaml

# Évaluation
python -m slmlab.cli.evaluate run model-base runs/adapter data/eval/heldout.jsonl
```