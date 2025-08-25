import typer
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from slmlab.utils.config import load_config
from slmlab.train.sft_lora import train as train_sft_lora

app = typer.Typer()

@app.command()
def run(cfg_path: Path = Path("configs/default.yaml")):
    cfg = load_config(cfg_path)
    tok = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True, trust_remote_code=True)
    ds = load_dataset("json", data_files={"train": cfg.paths.train, "eval": cfg.paths.eval})

    mode = cfg.get("templating", {}).get("mode", "base")
    method = cfg.method.get("method", "sft_lora")
    outdir = cfg.paths.out

    if mode == "chat":
        def tok_fn(batch):
            txts = [
                tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
                for ex in batch
            ]
            return tok(txts, text_target=[ex["messages"][-1]["content"] for ex in batch],
                       truncation=True, max_length=cfg.train.get("max_length", 1024))
    else:
        def tok_fn(batch):
            return tok(batch["prompt"], text_target=batch["label"],
                       truncation=True, max_length=cfg.train.get("max_length", 1024))

    cols = ds["train"].column_names
    ds_tok = ds.map(tok_fn, batched=True, num_proc=cfg.train.get("num_proc", 1), remove_columns=cols)

    
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if method == "sft_lora":
        train_sft_lora(cfg, ds_tok, outdir)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
if __name__ == "__main__":
    typer.run(run)
