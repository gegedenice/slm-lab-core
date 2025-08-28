# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# Modified for slm-lab project.

import json
import os
import sys
from dataclasses import dataclass, field

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logger = logging.get_logger(__name__)

@dataclass
class ScriptArguments:
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_config: str = field(default=None, metadata={"help": "the dataset config name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the dataset train split"})
    dataset_test_split: str = field(default="test", metadata={"help": "the dataset test split"})

def main(script_args, model_args, training_args):
    # Model init kwargs & Tokenizer
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True)

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.do_eval else None,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        dataset_text_field=training_args.dataset_text_field,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ModelConfig, SFTConfig))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        script_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        script_args, model_args, training_args = parser.parse_args_into_dataclasses()

    main(script_args, model_args, training_args)
