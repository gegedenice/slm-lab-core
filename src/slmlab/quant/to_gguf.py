# pseudo-code
import subprocess, typer, shutil
from pathlib import Path

app = typer.Typer()

@app.command()
def run(hf_model_dir: str, out_dir: str = "artifacts/gguf", q: str = "q4_k_m"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # suppose llama.cpp cloné dans tools/llama.cpp
    cmd = [
        "tools/llama.cpp/convert-hf-to-gguf.py",
        hf_model_dir, "--outfile", f"{out_dir}/model.gguf", "--outtype", q
    ]
    subprocess.check_call(["python", *cmd])
    typer.echo(f"GGUF écrit dans {out_dir}")

if __name__ == "__main__":
    app()