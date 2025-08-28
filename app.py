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

def get_config(use_case):
    if not use_case:
        return ""
    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    if not config_path.exists():
        return "Config file not found."
    with open(config_path, "r") as f:
        return f.read()

def save_config(use_case, config_text):
    if not use_case:
        return "Please select a use case first."
    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    with open(config_path, "w") as f:
        f.write(config_text)
    return f"Configuration for {use_case} saved."

def train(use_case):
    if not use_case:
        return "Please select a use case first."

    # This is a simplified example. In a real scenario, you would
    # handle the process asynchronously and stream the logs.
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

    config_editor = gr.Code(label="Config (default.yaml)", language="yaml", lines=20)

    with gr.Row():
        save_button = gr.Button("Save Config")
        train_button = gr.Button("Train")

    output_log = gr.Textbox(label="Training Log", lines=20, interactive=False)

    def update_config_editor(use_case):
        return get_config(use_case)

    use_case_dropdown.change(fn=update_config_editor, inputs=use_case_dropdown, outputs=config_editor)
    save_button.click(fn=save_config, inputs=[use_case_dropdown, config_editor], outputs=output_log)
    train_button.click(fn=train, inputs=use_case_dropdown, outputs=output_log)

    def refresh_use_cases():
        return gr.update(choices=get_use_cases())

    refresh_button.click(fn=refresh_use_cases, inputs=None, outputs=use_case_dropdown)

if __name__ == "__main__":
    demo.launch()
