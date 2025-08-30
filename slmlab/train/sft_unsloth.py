from unsloth import FastModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

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

def train(cfg, output_dir: str):
    # ---- 1. Load Model and Tokenizer ----
    model_name = _get_in(cfg, ["model", "name"])
    unsloth_settings = _get_in(cfg, ["method", "unsloth_settings"], {})

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=_get(unsloth_settings, "max_seq_length", 2048),
        dtype=None, # Let unsloth decide
        load_in_4bit=_get(unsloth_settings, "load_in_4bit", False),
    )
    model = model.to("cuda")

    # ---- 2. Apply PEFT ----
    peft_config = _get_in(cfg, ["method", "peft"], {})
    model = FastModel.get_peft_model(
        model,
        r=_get(peft_config, "r", 16),
        lora_alpha=_get(peft_config, "lora_alpha", 16),
        lora_dropout=_get(peft_config, "lora_dropout", 0),
        bias=_get(peft_config, "bias", "none"),
        random_state=_get(peft_config, "random_state", 3407),
        target_modules=_get(peft_config, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    )

    # ---- 3. Load and Format Dataset ----
    dataset_repo = _get_in(cfg, ["data", "repo"])
    if not dataset_repo:
        raise ValueError("data.repo not defined in config")

    dataset = load_dataset(dataset_repo, split="train")

    system_prompt = _get_in(cfg, ["templating", "prompts", "system_prompt"], "")

    def create_conversation(row):
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": row['answer']}
            ]
        }

    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

    dataset = dataset.map(formatting_prompts_func, remove_columns=['messages'], batched=True)

    # ---- 4. Configure and Run Trainer ----
    trainer_config = _get_in(cfg, ["method", "trainer"], {})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field=_get(trainer_config, "dataset_text_field", "text"),
            per_device_train_batch_size=_get(trainer_config, "per_device_train_batch_size", 2),
            gradient_accumulation_steps=_get(trainer_config, "gradient_accumulation_steps", 4),
            warmup_steps=_get(trainer_config, "warmup_steps", 5),
            max_steps=_get(trainer_config, "max_steps", 100),
            learning_rate=_get(trainer_config, "learning_rate", 5e-5),
            logging_steps=_get(trainer_config, "logging_steps", 1),
            optim=_get(trainer_config, "optim", "adamw_8bit"),
            weight_decay=_get(trainer_config, "weight_decay", 0.01),
            lr_scheduler_type=_get(trainer_config, "lr_scheduler_type", "linear"),
            seed=_get(trainer_config, "seed", 3407),
            output_dir=str(output_dir),
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    trainer.save_model(f"{str(output_dir).rstrip('/')}/adapter/")
