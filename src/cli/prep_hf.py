import typer
from pathlib import Path
from slmlab.io.loader_hf import build_jsonl_from_hf

app = typer.Typer()

@app.command()

def build(repo: str = "Geraldine/metadata-to-unimarc-reasoning",
          train: Path = Path("data/processed/train.jsonl"),
          eval: Path = Path("data/eval/heldout.jsonl"),
          eval_ratio: float = 0.2,
          seed: int = 42):
    ntr, nev = build_jsonl_from_hf(repo, train, eval, eval_ratio, seed)
    typer.echo(f"Wrote {ntr} train and {nev} eval examples")

if __name__ == "__main__":
    app()