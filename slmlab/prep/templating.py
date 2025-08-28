from typing import Dict, Any

def make_example(sample: dict, config: Any) -> Dict[str, Any]:
    """
    Creates a training example from a sample and a configuration object.
    The configuration object should contain the prompt templates.
    """
    mode = getattr(config.templating, "mode", "base")
    prompts = getattr(config.templating, "prompts", None)

    if not prompts:
        raise ValueError("No prompts found in the configuration.")

    if mode == "base":
        base_instruction = getattr(prompts, "base_instruction", None)
        if not base_instruction:
            raise ValueError("base_instruction not found in config.templating.prompts")

        # The sample should contain the keys to format the prompt
        prompt = base_instruction.format(**sample)
        return {"prompt": prompt, "label": sample.get("label", "")}

    elif mode == "chat":
        system_prompt = getattr(prompts, "system_prompt", "")

        # The sample is expected to be a list of messages
        messages = sample.get("messages", [])
        if not messages:
            raise ValueError("Chat mode requires a 'messages' key in the sample.")

        # If there's a system prompt in the config, and the sample doesn't
        # already have a system message, prepend it.
        has_system_message = any(msg.get("role") == "system" for msg in messages)
        if system_prompt and not has_system_message:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return {"messages": messages}

    else:
        raise ValueError(f"Unknown mode: {mode}")