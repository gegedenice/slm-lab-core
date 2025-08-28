import json
from pathlib import Path
import typer
from datasets import load_dataset
from slmlab.utils.config import load_config
from slmlab.prep.templating import make_example

app = typer.Typer()

@app.command()
def build_from_hf(use_case: str):
    """
    Builds a dataset from a Hugging Face repository as defined
    in the use-case configuration.
    """
    cfg = load_config(use_case)

    if not hasattr(cfg, "data") or not hasattr(cfg.data, "repo"):
        raise ValueError("data.repo not defined in the config.")

    repo = cfg.data.repo
    prompt_cols = getattr(cfg.data, "prompt_cols", [])
    label_col = getattr(cfg.data, "label_col", "label")
    eval_ratio = getattr(cfg.data, "eval_ratio", 0.2)

    ds = load_dataset(repo)["train"]

    # Filter rows that have all the required columns
    required_cols = prompt_cols + [label_col]
    rows = [r for r in ds if all(c in r and r[c] for c in required_cols)]

    n_eval = max(1, int(len(rows) * eval_ratio))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    use_case_dir = Path(f"use_cases/{use_case}")
    train_path = use_case_dir / cfg.paths.train
    eval_path = use_case_dir / cfg.paths.eval

    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as ftr:
        for r in train_rows:
            sample = {col: r[col] for col in prompt_cols}
            sample["label"] = r[label_col]
            ex = make_example(sample, cfg)
            ftr.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as fev:
        for r in eval_rows:
            sample = {col: r[col] for col in prompt_cols}
            sample["label"] = r[label_col]
            ex = make_example(sample, cfg)
            fev.write(json.dumps(ex, ensure_ascii=False) + "\n")

    typer.echo(f"Wrote {len(train_rows)} train and {len(eval_rows)} eval examples for use-case '{use_case}'.")

if __name__ == "__main__":
    app()
