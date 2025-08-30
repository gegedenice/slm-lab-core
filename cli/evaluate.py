import json
import typer
from pathlib import Path
from slmlab.utils.config import load_config
from slmlab.eval.runner import evaluate_models

app = typer.Typer()

def _get(obj, key, default=None):
    """Safe get for dict or SimpleNamespace."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _get_in(obj, keys, default=None):
    cur = obj
    for k in keys:
        cur = _get(cur, k, None)
        if cur is None:
            return default
    return cur

@app.command()
def run(use_case: str):
    """
    Runs evaluation for a given use-case.
    """
    cfg = load_config(use_case)

    use_case_dir = Path(f"use_cases/{use_case}")

    # Get baseline model name from the main model config
    baseline_model = _get_in(cfg, ["model", "name"])
    if not baseline_model:
        raise ValueError("model.name not found in config")

    # Construct paths from config
    eval_path = use_case_dir / _get_in(cfg, ["paths", "eval"])
    output_dir = use_case_dir / _get_in(cfg, ["paths", "out"])

    # The 'tuned' model is the adapter from the output directory
    tuned_model_path = output_dir / "adapter"

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    if not tuned_model_path.exists():
        raise FileNotFoundError(f"Tuned model adapter not found: {tuned_model_path}")

    typer.echo(f"Evaluating use-case: {use_case}")
    typer.echo(f"  - Baseline model: {baseline_model}")
    typer.echo(f"  - Tuned adapter: {tuned_model_path}")
    typer.echo(f"  - Evaluation data: {eval_path}")

    report = evaluate_models(baseline_model, str(tuned_model_path), eval_path)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    typer.echo(f"\nReport saved to {report_path}")
    # Pretty print main scores
    typer.echo("\n--- Key Scores ---")
    for model_type, scores in report.get("scores", {}).items():
        typer.echo(f"  {model_type.capitalize()}:")
        typer.echo(f"    - Exact Match: {scores.get('exact', 0):.4f}")
        typer.echo(f"    - ROUGE-L:     {scores.get('rougeL', 0):.4f}")
        typer.echo(f"    - BERTScore F1:{scores.get('bertscore_f1', 0):.4f}")


if __name__ == "__main__":
    app()