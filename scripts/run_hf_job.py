import sys
import json
import subprocess
import typer
from slmlab.utils.config import load_config

app = typer.Typer()

def _to_dict(ns):
    if isinstance(ns, dict):
        return {k: _to_dict(v) for k, v in ns.items()}
    elif hasattr(ns, '__dict__'):
        return {k: _to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [_to_dict(i) for i in ns]
    else:
        return ns

@app.command()
def main(use_case: str):
    """
    Runs a Hugging Face Job for the given use-case.
    """
    cfg = load_config(use_case)

    if not hasattr(cfg, "hf_job"):
        raise ValueError(f"hf_job configuration not found for use-case '{use_case}'")

    hf_job_cfg = cfg.hf_job

    # Prepare the config for the SFT script
    sft_config = {
        "script_args": _to_dict(hf_job_cfg.script_args),
        "model_args": _to_dict(hf_job_cfg.model_args),
        "training_args": _to_dict(hf_job_cfg.training_args),
    }

    # Create a temporary JSON file for the config
    tmp_config_path = f"/tmp/{use_case}_hf_job_config.json"
    with open(tmp_config_path, "w") as f:
        json.dump(sft_config, f, indent=2)

    instance_type = getattr(hf_job_cfg, "instance_type", "cpu-upgrade")

    command = [
        "huggingface-cli", "job", "run",
        "--instance-type", instance_type,
        "scripts/hf_sft_job.py",
        "--",
        tmp_config_path
    ]

    typer.echo(f"Running command: {' '.join(command)}")

    subprocess.run(command, check=True)

if __name__ == "__main__":
    app()
