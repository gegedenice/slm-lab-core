import gradio as gr
import os
import yaml
from pathlib import Path
import subprocess
import numbers
from collections.abc import MutableMapping

# --- Constants ---
MAX_PARAMS_PER_SECTION = 20

# --- Utility Functions ---

def get_use_cases():
    use_cases_dir = "use_cases"
    if not os.path.exists(use_cases_dir): return []
    return sorted([d for d in os.listdir(use_cases_dir) if os.path.isdir(os.path.join(use_cases_dir, d))])

def get_method_files(use_case):
    if not use_case: return []
    methods_dir = Path(f"use_cases/{use_case}/configs/methods")
    if not methods_dir.exists(): return []
    return sorted([p.name for p in methods_dir.glob("*.yaml")])

def load_config(use_case):
    if not use_case: return {}
    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    if not config_path.exists(): return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config(use_case, config_data):
    if not use_case: return "No use-case selected."
    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, indent=2)
    return f"Config for '{use_case}' saved successfully."

def save_section(use_case, section_name, *values):
    """Saves a single section back to the default.yaml."""
    config_data = load_config(use_case)
    section_data = config_data.get(section_name, {})

    keys = list(section_data.keys())
    for i, key in enumerate(keys):
        original_value = section_data.get(key)
        try:
            if isinstance(original_value, bool):
                new_value = bool(values[i])
            elif isinstance(original_value, numbers.Number):
                new_value = type(original_value)(values[i])
            else:
                new_value = values[i]
            section_data[key] = new_value
        except (ValueError, TypeError):
            section_data[key] = values[i]

    config_data[section_name] = section_data
    return save_config(use_case, config_data)

def run_make_command(use_case, command):
    if not use_case: return "Please select a use case first."
    process = subprocess.Popen(["make", command, f"USE_CASE={use_case}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output
    process.stdout.close()
    if process.wait() != 0: yield output + f"\n\nCommand '{command}' failed"

# --- UI Generation ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SLM Lab - Full Control Panel")

    config_data_state = gr.State({})

    with gr.Row():
        use_case_dropdown = gr.Dropdown(choices=get_use_cases(), value=get_use_cases()[0] if get_use_cases() else None, label="Select Use Case")
        refresh_button = gr.Button("Refresh")

    with gr.Tabs():
        with gr.TabItem("Method"):
            method_dropdown = gr.Dropdown(label="Select Method File")
            save_method_button = gr.Button("Save Method Selection")

        # Create static containers for each section
        section_tabs = {}
        for section_name in ["data", "templating", "train", "hf_job"]:
            with gr.TabItem(section_name.capitalize()):
                components = []
                for i in range(MAX_PARAMS_PER_SECTION):
                    components.append(gr.Textbox(visible=False, label=f"param_{i}"))
                save_button = gr.Button(f"Save {section_name.capitalize()} Section")
                section_tabs[section_name] = {"components": components, "save_button": save_button}

        with gr.TabItem("Raw YAML"):
            raw_yaml_editor = gr.Code(label="default.yaml", language="yaml", lines=30)
            save_raw_yaml_button = gr.Button("Save Raw YAML")

    with gr.Row():
        prep_button = gr.Button("Prepare Data")
        train_button = gr.Button("Train Use Case")
        train_hf_job_button = gr.Button("Launch HF Job")

    output_log = gr.Textbox(label="Log Output", lines=15, interactive=False)

    # --- Event Handlers ---

    def update_all_uis(use_case):
        config_data = load_config(use_case)

        # Method Tab
        methods = get_method_files(use_case)
        selected_method = config_data.get("method", "N/A")
        method_ui = gr.update(choices=methods, value=selected_method)

        # Raw YAML
        raw_yaml = yaml.dump(config_data, sort_keys=False, indent=2)

        # Structured Tabs
        outputs = [config_data, method_ui, raw_yaml]
        for section_name, tab_info in section_tabs.items():
            section_data = config_data.get(section_name, {})
            keys = list(section_data.keys())
            for i in range(MAX_PARAMS_PER_SECTION):
                if i < len(keys):
                    key = keys[i]
                    value = section_data[key]
                    if isinstance(value, bool):
                        outputs.append(gr.Checkbox(value=value, label=key, visible=True))
                    else:
                        outputs.append(gr.Textbox(value=str(value), label=key, visible=True))
                else:
                    outputs.append(gr.Textbox(visible=False))
        return outputs

    # Wire up save buttons for structured tabs
    for section_name, tab_info in section_tabs.items():
        tab_info["save_button"].click(
            fn=lambda uc, s=section_name, *vals: save_section(uc, s, *vals),
            inputs=[use_case_dropdown] + tab_info["components"],
            outputs=output_log
        )

    # Method Save Logic
    def save_method(use_case, selected_method_file):
        config_data = load_config(use_case)
        config_data['method'] = selected_method_file
        return save_config(use_case, config_data)

    save_method_button.click(save_method, inputs=[use_case_dropdown, method_dropdown], outputs=output_log)

    # Raw YAML Save Logic
    def save_raw_yaml(use_case, raw_text):
        try:
            yaml.safe_load(raw_text)
            config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            return f"Successfully saved default.yaml for {use_case}."
        except yaml.YAMLError as e:
            return f"Invalid YAML: {e}"

    save_raw_yaml_button.click(save_raw_yaml, inputs=[use_case_dropdown, raw_yaml_editor], outputs=output_log)

    # Main UI update logic
    all_structured_components = [comp for info in section_tabs.values() for comp in info["components"]]
    outputs_for_update = [config_data_state, method_dropdown, raw_yaml_editor] + all_structured_components

    use_case_dropdown.change(fn=update_all_uis, inputs=use_case_dropdown, outputs=outputs_for_update)
    refresh_button.click(fn=update_all_uis, inputs=use_case_dropdown, outputs=outputs_for_update)
    demo.load(fn=update_all_uis, inputs=use_case_dropdown, outputs=outputs_for_update)

    # Action buttons
    prep_button.click(lambda uc: run_make_command(uc, "prep"), inputs=use_case_dropdown, outputs=output_log)
    train_button.click(lambda uc: run_make_command(uc, "train"), inputs=use_case_dropdown, outputs=output_log)
    train_hf_job_button.click(lambda uc: run_make_command(uc, "train-hf-job"), inputs=use_case_dropdown, outputs=output_log)

if __name__ == "__main__":
    demo.launch()
