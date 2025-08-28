import json
from pathlib import Path
import typer
from datasets import load_dataset
import slmlab
from slmlab.utils.config import load_config
from slm_lab_unimarc.prep.templating import make_example

def get_mode_from_cfg(cfg, default="base"):
    # Works for both dict and SimpleNamespace trees
    if isinstance(cfg, dict):
        return cfg.get("templating", {}).get("mode", default)
    templating = getattr(cfg, "templating", None)
    if templating is None:
        return default
    return getattr(templating, "mode", default)

app = typer.Typer()

@app.command()
def build(
    repo: str = typer.Option(..., "--repo", "-r", help="HF dataset repo, e.g. owner/dataset"),
    train: Path = typer.Option(Path("data/processed/train.jsonl"), "--train"),
    eval: Path = typer.Option(Path("data/eval/heldout.jsonl"), "--eval"),
    eval_ratio: float = typer.Option(0.2, "--eval-ratio", min=0.0, max=1.0),
    cfg_path: Path = typer.Option(Path("configs/default.yaml"), "--cfg-path"),
):
    # Note: load_config from slmlab now expects a use_case name.
    # This script is use-case specific, so we can't use the generic one directly.
    # We will assume for now that this script is run from the use-case directory.
    with open(cfg_path, "r") as f:
        import yaml
        cfg = yaml.safe_load(f)

    mode = get_mode_from_cfg(cfg, default="base")
    ds = load_dataset(repo)["train"]
    rows = [r for r in ds if r.get("metadata") and r.get("unimarc_record")]
    n_eval = max(1, int(len(rows) * eval_ratio))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    train.parent.mkdir(parents=True, exist_ok=True)
    eval.parent.mkdir(parents=True, exist_ok=True)

    with train.open("w", encoding="utf-8") as ftr:
        for r in train_rows:
            ex = make_example(r["metadata"], r["unimarc_record"], mode=mode)
            ftr.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with eval.open("w", encoding="utf-8") as fev:
        for r in eval_rows:
            ex = make_example(r["metadata"], r["unimarc_record"], mode=mode)
            fev.write(json.dumps(ex, ensure_ascii=False) + "\n")

    typer.echo(f"Wrote {len(train_rows)} train and {len(eval_rows)} eval examples (mode={mode}).")

if __name__ == "__main__":
    typer.run(build)
