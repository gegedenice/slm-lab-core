import json, typer
from pathlib import Path
from slmlab.eval.runner import evaluate_models

app = typer.Typer()

@app.command()
def run(baseline: str, tuned: str, eval_path: Path = Path("data/eval/heldout.jsonl")):
    report = evaluate_models(baseline, tuned, eval_path)
    out = Path("runs/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Report saved to {out}")

if __name__ == "__main__":
    app()