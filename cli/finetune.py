# cli/finetune.py
import typer
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from slmlab.utils.config import load_config
from slmlab.train.sft_lora import train as train_sft_lora

app = typer.Typer()

def _get(obj, key, default=None):
    """Safe get for dict or SimpleNamespace."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

@app.command()
def run(cfg_path: Path = Path("configs/default.yaml")):
    cfg = load_config(cfg_path)

    model_name = _get(_get(cfg, "model"), "name")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    train_path = _get(_get(cfg, "paths"), "train")
    eval_path  = _get(_get(cfg, "paths"), "eval")
    outdir     = _get(_get(cfg, "paths"), "out")

    ds = load_dataset("json", data_files={"train": train_path, "eval": eval_path})

    mode   = _get(_get(cfg, "templating"), "mode", "base")
    method = _get(_get(cfg, "method"), "method", "sft_lora")

    max_len = _get(_get(cfg, "train"), "max_length", 1024)
    num_proc = _get(_get(cfg, "train"), "num_proc", 1)

    if mode == "chat":
        def tok_fn(batch):
            txts = [
                tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
                for ex in batch
            ]
            labels = [ex["messages"][-1]["content"] for ex in batch]
            return tok(txts, text_target=labels, truncation=True, max_length=max_len)
    else:
        def tok_fn(batch):
            return tok(batch["prompt"], text_target=batch["label"],
                       truncation=True, max_length=max_len)

    cols = ds["train"].column_names
    ds_tok = ds.map(tok_fn, batched=True, num_proc=num_proc, remove_columns=cols)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    if method == "sft_lora":
        train_sft_lora(cfg, ds_tok, outdir)
    else:
        raise ValueError(f"Unsupported method: {method}")

if __name__ == "__main__":
    typer.run(run)
