import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict
import torch.nn as nn

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# ---- helpers: work with dicts or SimpleNamespace trees ----
def _get(obj, key, default=None):
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

# ---- helpers: peft modules parameters ----   
def _count_trainable_params(m):
    return sum(p.requires_grad for p in m.parameters())
    
def _num_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _guess_target_modules(model):
    # Return unique module names containing Linear layers under attention/MLP blocks
    names = []
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            # keep short leaf name
            leaf = n.split(".")[-1]
            names.append(leaf)
    return sorted(set(names))

def train(cfg, ds_tokenized, output_dir: str):
    model_name = _get_in(cfg, ["model", "name"])
    trust_remote_code = _get_in(cfg, ["model", "trust_remote_code"], True)

    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = get_device()
    print(f"[slmlab] Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    ).to(device)
    
    # recommended when using gradient checkpointing
    model.config.use_cache = False

    # LoRA config (handle dict or namespace)
    peft_conf = _get(cfg, "method", {})  # may be ns
    peft_conf = _get(peft_conf, "peft", {})  # dive into 'peft' sub-config if present
    tmods = _get(peft_conf, "target_modules", None)  # from your cfg helper
    if tmods in (None, [], ""):
        tmods = "all-linear"
        
    lora = LoraConfig(
        r=_get(peft_conf, "r", 16),
        lora_alpha=_get(peft_conf, "lora_alpha", 32),
        lora_dropout=_get(peft_conf, "lora_dropout", 0.05),
        target_modules=tmods,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    
    print("[peft] trainable keys (first 20):", list(get_peft_model_state_dict(model).keys())[:20])
    print("[peft] #trainable params:", _num_trainable(model))

    if _num_trainable(model) == 0:
        # Fallback: target all linear-like layers by leaf name
        leaf_linear_names = sorted({n.split(".")[-1]
            for n, mod in model.named_modules() if isinstance(mod, nn.Linear)})
        raise RuntimeError(
            "LoRA attached to 0 params. Your `target_modules` didn't match.\n"
            f"Try `target_modules: \"all-linear\"` or one of these leaf names:\n{leaf_linear_names}"
        )
    
    if _count_trainable_params(model) == 0:
        # fallback if user provided names matched nothing
        lora.target_modules = "all-linear"
        model = get_peft_model(model, lora)  # rewrap

    # TrainingArguments — read safely and coerce types where needed
    train_cfg = _get(cfg, "train", {})
    lr = _get(train_cfg, "lr", 2e-4)
    if isinstance(lr, str):
        try:
            lr = float(lr)
        except ValueError:
            lr = 2e-4  # fallback

    targs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=_get(train_cfg, "per_device_train_batch_size", 1),
        gradient_accumulation_steps=_get(train_cfg, "gradient_accumulation_steps", 1),
        learning_rate=lr,
        logging_steps=_get(train_cfg, "logging_steps", 50),
        eval_steps=_get(train_cfg, "eval_steps", 200),
        save_steps=_get(train_cfg, "save_steps", 200),
        warmup_ratio=_get(train_cfg, "warmup_ratio", 0.0),
        lr_scheduler_type=_get(train_cfg, "lr_scheduler_type", "linear"),
        seed=_get(cfg, "seed", 42),
        fp16=_get(train_cfg, "fp16", False),
        bf16=_get(train_cfg, "bf16", False),
        report_to=["none"],
        gradient_checkpointing=_get(train_cfg, "gradient_checkpointing", False),
    )

    max_steps = _get(train_cfg, "max_steps", None)
    if max_steps:
        targs["max_steps"] = max_steps
    else:
        targs["num_train_epochs"] = _get(train_cfg, "num_train_epochs", 1)

    args = TrainingArguments(**targs)

    # DatasetDict is dict-like; .get is fine, but use [] fallback if not present
    eval_ds = ds_tokenized.get("eval") if hasattr(ds_tokenized, "get") else ds_tokenized["eval"] if "eval" in ds_tokenized else None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=eval_ds,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(f"{output_dir.rstrip('/')}/adapter/")
