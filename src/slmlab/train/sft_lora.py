# src/slmlab/train/sft_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def train(cfg, ds_tokenized, output_dir: str):
    tok = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True, trust_remote_code=cfg.model.get("trust_remote_code", True))
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, trust_remote_code=cfg.model.get("trust_remote_code", True))
    peft_conf = cfg.method.get("peft", {})
    lora = LoraConfig(
        r=peft_conf.get("r", 16),
        lora_alpha=peft_conf.get("lora_alpha", 32),
        lora_dropout=peft_conf.get("lora_dropout", 0.05),
        target_modules=peft_conf.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    )
    model = get_peft_model(model, lora)

    targs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.lr,
        logging_steps=cfg.train.logging_steps,
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        warmup_ratio=cfg.train.get("warmup_ratio", 0.0),
        lr_scheduler_type=cfg.train.get("lr_scheduler_type", "linear"),
        seed=cfg.get("seed", 42),
        fp16=cfg.train.get("fp16", False),
        bf16=cfg.train.get("bf16", False),
        report_to=["none"],
        gradient_checkpointing=cfg.train.get("gradient_checkpointing", False),
    )
    if cfg.train.get("max_steps", None):
        targs["max_steps"] = cfg.train.max_steps
    else:
        targs["num_train_epochs"] = cfg.train.get("num_train_epochs", 1)

    args = TrainingArguments(**targs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized.get("eval"),
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(output_dir + "/adapter/")
