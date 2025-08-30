import gradio as gr
import os
import yaml
from pathlib import Path
import subprocess
from glob import glob
import numbers

# --- Data Functions ---

def get_use_cases():
    """Returns a list of available use-case directories."""
    use_cases_dir = "use_cases"
    if not os.path.exists(use_cases_dir):
        return []
    return sorted([d for d in os.listdir(use_cases_dir) if os.path.isdir(os.path.join(use_cases_dir, d))])

def get_config_files(use_case):
    """Returns a list of all yaml config files for a given use-case."""
    if not use_case:
        return []
    config_dir = Path(f"use_cases/{use_case}/configs")
    return sorted([str(p) for p in config_dir.rglob("*.yaml")])

def load_file_content(filepath):
    """Loads the raw text content of a file."""
    if not filepath or not Path(filepath).exists():
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def save_file_content(filepath, content):
    """Saves content to a file."""
    if not filepath:
        return "Error: No file path specified."
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved {Path(filepath).name}."
    except Exception as e:
        return f"Error saving file: {e}"

def save_train_params(use_case, keys, *values):
    """Saves the structured training parameters back to default.yaml."""
    if not use_case:
        return "No use-case selected."

    config_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if "train" not in config_data:
        config_data["train"] = {}

    for key, value in zip(keys, values):
        # Attempt to cast back to original type (e.g., int, float)
        original_value = config_data["train"].get(key)
        try:
            if isinstance(original_value, bool):
                new_value = str(value).lower() in ['true', '1', 't', 'y', 'yes']
            elif isinstance(original_value, numbers.Number):
                new_value = type(original_value)(value)
            else:
                new_value = value
            config_data["train"][key] = new_value
        except (ValueError, TypeError):
            config_data["train"][key] = value # Save as string if cast fails

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, indent=2)

    return f"Training parameters for '{use_case}' saved."


def train(use_case):
    # (train function remains the same)
    if not use_case:
        return "Please select a use case first."
    process = subprocess.Popen(["make", "train", f"USE_CASE={use_case}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output
    process.stdout.close()
    if process.wait() != 0:
        yield output + "\n\nTraining failed"

# --- App Initialization & UI Generation ---
available_use_cases = get_use_cases()
selected_use_case = available_use_cases[0] if available_use_cases else None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SLM Lab")

    config_file_paths_state = gr.State([])
    train_params_keys_state = gr.State([])
    selected_tab_index_state = gr.State(0) # New: State to hold the selected tab index

    with gr.Row():
        use_case_dropdown = gr.Dropdown(choices=available_use_cases, value=selected_use_case, label="Select Use Case")
        refresh_button = gr.Button("Refresh")

    main_tabs = gr.Tabs() # Placeholder for dynamic tabs


    with gr.Row():
        train_button = gr.Button("Train Use Case")

    output_log = gr.Textbox(label="Log Output", lines=15, interactive=False)

    # --- UI Generation and Event Handlers ---

    MAX_FILE_EDITORS = 10
    MAX_TRAIN_PARAMS = 20

    def create_main_ui(use_case):
        config_files = get_config_files(use_case)
        file_editors = []
        train_params_ui = []
        train_params_keys = []

        with gr.Tabs() as new_tabs:
            # Create a tab for the structured training parameters
            with gr.TabItem("Structured Train Config"):
                default_yaml_path = f"use_cases/{use_case}/configs/default.yaml"
                default_yaml_content = load_file_content(default_yaml_path)
                train_config = yaml.safe_load(default_yaml_content).get("train", {})

                with gr.Column():
                    for key, value in train_config.items():
                        train_params_keys.append(key)
                        train_params_ui.append(gr.Textbox(value=str(value), label=key))

                save_train_button = gr.Button("Save Training Params")

            # Create tabs for each raw YAML file
            for filepath in config_files:
                p = Path(filepath)
                label = f"{p.parent.name}/{p.name}" if p.parent.name != "configs" else p.name
                with gr.TabItem(label):
                    content = load_file_content(filepath)
                    editor = gr.Code(value=content, language="yaml", lines=25)
                    file_editors.append(editor)

        save_file_button = gr.Button("Save Active YAML File")

        # Pad lists to ensure consistent output length for Gradio
        padded_file_editors = file_editors + [None] * (MAX_FILE_EDITORS - len(file_editors))
        padded_train_params_ui = train_params_ui + [None] * (MAX_TRAIN_PARAMS - len(train_params_ui))

        # Return values for all output components, including initial selected tab index
        return new_tabs, config_files, train_params_keys, save_file_button, save_train_button, 0, *padded_file_editors, *padded_train_params_ui

    # Define all potential output components (placeholders for dynamic UI)
    dummy_file_editors = [gr.Code(visible=False) for _ in range(MAX_FILE_EDITORS)]
    dummy_train_params = [gr.Textbox(visible=False) for _ in range(MAX_TRAIN_PARAMS)]
    dummy_save_file_button = gr.Button(visible=False)
    dummy_save_train_button = gr.Button(visible=False)

    # Wire up the dropdown change event
    use_case_dropdown.change(
        fn=create_main_ui,
        inputs=use_case_dropdown,
        outputs=[main_tabs, config_file_paths_state, train_params_keys_state, dummy_save_file_button, dummy_save_train_button, selected_tab_index_state] + dummy_file_editors + dummy_train_params
    )

    # Wire up the save buttons (click events attached to placeholders)
    dummy_save_file_button.click(
        fn=lambda uc, files, tab_index, *contents:
            save_file_content(files[int(tab_index) - 1], contents[int(tab_index) - 1])
            if tab_index > 0 and (int(tab_index) - 1) < len(files) else "Cannot save structured config as a raw file.",
        inputs=[use_case_dropdown, config_file_paths_state, selected_tab_index_state] + dummy_file_editors,
        outputs=output_log
    )
    dummy_save_train_button.click(
        fn=save_train_params,
        inputs=[use_case_dropdown, train_params_keys_state] + dummy_train_params,
        outputs=output_log
    )

    # Wire up other buttons
    train_button.click(fn=train, inputs=use_case_dropdown, outputs=output_log)
    refresh_button.click(
        fn=lambda: (gr.update(choices=get_use_cases(), value=get_use_cases()[0] if get_use_cases() else None)),
        inputs=None,
        outputs=use_case_dropdown
    ).then(
        fn=create_main_ui,
        inputs=use_case_dropdown,
        outputs=[main_tabs, config_file_paths_state, train_params_keys_state, dummy_save_file_button, dummy_save_train_button, selected_tab_index_state] + dummy_file_editors + dummy_train_params
    )

    # New: Add a handler to update selected_tab_index_state when a tab is selected
    main_tabs.select(
        fn=lambda evt: evt.index, # evt.index is the 0-indexed selected tab position
        inputs=None,
        outputs=selected_tab_index_state
    )

    # Initial load of the UI
    demo.load(
        fn=create_main_ui,
        inputs=gr.State(selected_use_case), # Pass the initial selected_use_case as a state
        outputs=[main_tabs, config_file_paths_state, train_params_keys_state, dummy_save_file_button, dummy_save_train_button, selected_tab_index_state] + dummy_file_editors + dummy_train_params
    )

if __name__ == "__main__":
    demo.launch(share=True)
