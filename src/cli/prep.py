import json, typer
from pathlib import Path
from slmlab.prep.templating import apply_template


app = typer.Typer()


@app.command()
def build(in_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as f, out_path.open("w") as g:
        for line in f:
            sample = json.loads(line)
            ex = apply_template(sample)
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()