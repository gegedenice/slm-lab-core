# slm-lab

Boîte à outils pour le post-training de Small Language Models (SLM).

## Fonctionnalités
- Préparation de données (templates instruction & chat)
- Fine-tuning LoRA (CPU/GPU)
- Distillation (teacher → student)
- Évaluation (rouge, exact match, BERTScore, XML-validité…)
- Packaging & service API

---

## Installation
```bash
pip install -e .
```
---

## Structure du Projet

Le projet est structuré comme un "monorepo" qui contient la bibliothèque principale `slmlab` et un répertoire `use_cases` pour les différents cas d'usage.

```
slm-lab/
├─ README.md
├─ pyproject.toml
├─ Makefile
├─ slmlab/              # Bibliothèque principale
│  └─ ...
├─ cli/                 # Scripts en ligne de commande
│  └─ ...
├─ use_cases/           # Cas d'usage
│  └─ unimarc/
│     ├─ configs/
│     │  └─ default.yaml
│     ├─ data/
│     └─ scripts/
└─ app.py               # Interface Gradio
```

---

## Utilisation

### Ligne de commande

Le `Makefile` à la racine du projet fournit des commandes pour travailler avec les cas d'usage. Vous pouvez spécifier le cas d'usage avec la variable `USE_CASE`.

```bash
# Entraîner le cas d'usage 'unimarc' (par défaut)
make train

# Entraîner un autre cas d'usage
make train USE_CASE=mon_autre_cas
```

### Interface Gradio

Une interface Gradio est disponible pour gérer les configurations de manière interactive.

```bash
# Lancer l'interface
make run-gradio
```

Ouvrez votre navigateur à l'adresse [http://127.0.0.1:7860](http://127.0.0.1:7860).

---

## Créer un nouveau cas d'usage

Pour créer un nouveau cas d'usage, il suffit de copier un cas existant et de le modifier.

1.  Copiez `use_cases/unimarc` vers `use_cases/mon_nouveau_cas`.
2.  Modifiez les fichiers de configuration dans `use_cases/mon_nouveau_cas/configs`.
3.  Ajoutez vos données dans `use_cases/mon_nouveau_cas/data`.
4.  Lancez l'entraînement avec `make train USE_CASE=mon_nouveau_cas`.