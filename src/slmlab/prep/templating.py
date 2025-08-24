INSTRUCTION_TEMPLATES = {}

def register_template(name, func):
    INSTRUCTION_TEMPLATES[name] = func

def apply_template(sample: dict):
    task = sample["task"]
    if task not in INSTRUCTION_TEMPLATES:
        raise ValueError(f"No template for {task}")
    return INSTRUCTION_TEMPLATES[task](sample)