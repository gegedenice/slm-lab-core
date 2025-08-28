# cli/finetune.py
import typer
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from slmlab.utils.config import load_config
from slmlab.train.sft_lora import train as train_sft_lora
from slmlab.train.sft_unsloth import train as train_sft_unsloth

app = typer.Typer()

def _get(obj, key, default=None):
    """Safe get for dict or SimpleNamespace."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

@app.command()
def run(use_case: str):
    cfg = load_config(use_case)

    use_case_dir = Path(f"use_cases/{use_case}")
    outdir = use_case_dir / _get(_get(cfg, "paths"), "out")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    method = _get(_get(cfg, "method"), "method", "sft_lora")

    if method == "sft_lora":
        # sft_lora expects a tokenized dataset
        model_name = _get(_get(cfg, "model"), "name")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

        train_path = use_case_dir / _get(_get(cfg, "paths"), "train")
        eval_path  = use_case_dir / _get(_get(cfg, "paths"), "eval")

        ds = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})

        mode   = _get(_get(cfg, "templating"), "mode", "base")
        max_len = _get(_get(cfg, "train"), "max_length", 1024)
        num_proc = _get(_get(cfg, "train"), "num_proc", 1)

        if mode == "chat":
            def tok_fn(batch):
                msgs_list = batch["messages"]
                txts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in msgs_list]
                labels = [(msgs[-1]["content"] if msgs and msgs[-1].get("role") == "assistant" else "") for msgs in msgs_list]
                return tok(txts, text_target=labels, truncation=True, max_length=max_len)
        else:
            def tok_fn(batch):
                return tok(batch["prompt"], text_target=batch["label"], truncation=True, max_length=max_len)

        cols = ds["train"].column_names
        ds_tok = ds.map(tok_fn, batched=True, num_proc=num_proc, remove_columns=cols)

        train_sft_lora(cfg, ds_tok, outdir)

    elif method == "unsloth":
        # unsloth handles its own data loading and tokenization
        train_sft_unsloth(cfg, outdir)

    else:
        raise ValueError(f"Unsupported method: {method}")

if __name__ == "__main__":
    typer.run(run)
