import gradio as gr
import os
import yaml
from pathlib import Path
import subprocess

def get_use_cases():
    use_cases_dir = "use_cases"
    if not os.path.exists(use_cases_dir):
        return []
    return [d for d in os.listdir(use_cases_dir) if os.path.isdir(os.path.join(use_cases_dir, d))]

def get_config_details(use_case):
    if not use_case:
        return "", "", "", gr.update(visible=False)

    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    if not config_path.exists():
        return "Config file not found.", "", "", gr.update(visible=False)

    with open(config_path, "r") as f:
        config_text = f.read()

    config_data = yaml.safe_load(config_text) or {}

    prompts = config_data.get("templating", {}).get("prompts", {})
    base_instruction = prompts.get("base_instruction", "")
    system_prompt = prompts.get("system_prompt", "")

    prompts_visible = "prompts" in config_data.get("templating", {})

    return config_text, base_instruction, system_prompt, gr.update(visible=prompts_visible)

def save_config(use_case, config_text, base_instruction, system_prompt):
    if not use_case:
        return "Please select a use case first."

    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")

    config_data = yaml.safe_load(config_text) or {}

    if "prompts" in config_data.get("templating", {}):
        config_data["templating"]["prompts"]["base_instruction"] = base_instruction
        config_data["templating"]["prompts"]["system_prompt"] = system_prompt

    with open(config_path, "w") as f:
        yaml.dump(config_data, f, sort_keys=False, indent=2)

    return f"Configuration for {use_case} saved."

def train(use_case):
    if not use_case:
        return "Please select a use case first."

    process = subprocess.Popen(
        ["make", "train", f"USE_CASE={use_case}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output

    process.stdout.close()
    return_code = process.wait()
    if return_code:
        yield output + f"\n\nTraining failed with return code {return_code}"


with gr.Blocks() as demo:
    gr.Markdown("# SLM Lab")

    with gr.Row():
        use_case_dropdown = gr.Dropdown(choices=get_use_cases(), label="Select Use Case")
        refresh_button = gr.Button("Refresh")

    with gr.Tabs():
        with gr.TabItem("YAML Config"):
            config_editor = gr.Code(label="Config (default.yaml)", language="yaml", lines=20)
        with gr.TabItem("Prompt Templates"):
            with gr.Box(visible=False) as prompt_box:
                base_instruction_editor = gr.Textbox(label="Base Instruction", lines=10)
                system_prompt_editor = gr.Textbox(label="System Prompt", lines=5)

    with gr.Row():
        save_button = gr.Button("Save Config")
        train_button = gr.Button("Train")

    output_log = gr.Textbox(label="Training Log", lines=20, interactive=False)

    use_case_dropdown.change(
        fn=get_config_details,
        inputs=use_case_dropdown,
        outputs=[config_editor, base_instruction_editor, system_prompt_editor, prompt_box]
    )

    save_button.click(
        fn=save_config,
        inputs=[use_case_dropdown, config_editor, base_instruction_editor, system_prompt_editor],
        outputs=output_log
    )

    train_button.click(fn=train, inputs=use_case_dropdown, outputs=output_log)

    def refresh_use_cases():
        return gr.update(choices=get_use_cases())

    refresh_button.click(fn=refresh_use_cases, inputs=None, outputs=use_case_dropdown)

if __name__ == "__main__":
    demo.launch()
