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

    # Define main_tabs and its TabItems statically
    main_tabs = gr.Tabs()
    with main_tabs:
        with gr.TabItem("Structured Train Config") as structured_train_config_tab:
            # Placeholders for structured training parameters
            train_params_ui_placeholders = [gr.Textbox(visible=False, label=f"Param {i}") for i in range(20)] # MAX_TRAIN_PARAMS is 20
            save_train_button = gr.Button("Save Training Params", visible=False)

        file_tab_items = []
        file_editor_placeholders = []
        for i in range(10): # MAX_FILE_EDITORS is 10
            with gr.TabItem(f"File {i+1}", visible=False) as file_tab_item:
                file_tab_items.append(file_tab_item)
                file_editor_placeholders.append(gr.Code(language="yaml", lines=25, visible=False))
        save_file_button = gr.Button("Save Active YAML File", visible=False)

    with gr.Row():
        train_button = gr.Button("Train Use Case")

    output_log = gr.Textbox(label="Log Output", lines=15, interactive=False)

    # --- UI Generation and Event Handlers ---

    MAX_FILE_EDITORS = 10
    MAX_TRAIN_PARAMS = 20

    def create_main_ui(use_case):
        # Initialize all updates to invisible
        updates = {
            structured_train_config_tab: gr.TabItem(visible=False),
            save_train_button: gr.Button(visible=False)
        }
        for i in range(MAX_TRAIN_PARAMS):
            updates[train_params_ui_placeholders[i]] = gr.Textbox(visible=False)
        for i in range(MAX_FILE_EDITORS):
            updates[file_tab_items[i]] = gr.TabItem(visible=False)
            updates[file_editor_placeholders[i]] = gr.Code(visible=False)
        updates[save_file_button] = gr.Button(visible=False)

        if use_case:
            # Structured Train Config Tab
            updates[structured_train_config_tab] = gr.TabItem(visible=True)
            default_yaml_path = f"use_cases/{use_case}/configs/default.yaml"
            default_yaml_content = load_file_content(default_yaml_path)
            train_config = yaml.safe_load(default_yaml_content).get("train", {})
            train_params_keys = list(train_config.keys())

            for i, (key, value) in enumerate(train_config.items()):
                updates[train_params_ui_placeholders[i]] = gr.Textbox(value=str(value), label=key, visible=True)
            updates[save_train_button] = gr.Button(visible=True)

            # Raw YAML File Tabs
            config_files = get_config_files(use_case)
            # Filter out default.yaml to avoid showing it twice
            raw_yaml_files = [f for f in config_files if Path(f).name != "default.yaml"]

            for i, filepath in enumerate(raw_yaml_files):
                if i < MAX_FILE_EDITORS:
                    p = Path(filepath)
                    label = f"{p.parent.name}/{p.name}" if p.parent.name != "configs" else p.name
                    content = load_file_content(filepath)
                    updates[file_tab_items[i]] = gr.TabItem(label=label, visible=True)
                    updates[file_editor_placeholders[i]] = gr.Code(value=content, language="yaml", lines=25, visible=True)

            if raw_yaml_files:
                updates[save_file_button] = gr.Button(visible=True)

            return (
                updates[structured_train_config_tab], # Output for structured_train_config_tab
                *([updates[p] for p in train_params_ui_placeholders]), # Outputs for train_params_ui_placeholders
                updates[save_train_button], # Output for save_train_button
                *[updates[t] for t in file_tab_items], # Outputs for file_tab_items
                *[updates[e] for e in file_editor_placeholders], # Outputs for file_editor_placeholders
                updates[save_file_button], # Output for save_file_button
                raw_yaml_files, # config_file_paths_state
                train_params_keys, # train_params_keys_state
                0 # selected_tab_index_state (initially select the first tab)
            )
        else:
            # If no use case, hide everything
            return (
                updates[structured_train_config_tab],
                *([updates[p] for p in train_params_ui_placeholders]),
                updates[save_train_button],
                *[updates[t] for t in file_tab_items],
                *[updates[e] for e in file_editor_placeholders],
                updates[save_file_button],
                [], # config_file_paths_state
                [], # train_params_keys_state
                0 # selected_tab_index_state
            )

    # Define all potential output components (placeholders for dynamic UI)
    all_tab_outputs = [
        structured_train_config_tab,
        *train_params_ui_placeholders,
        save_train_button,
        *file_tab_items,
        *file_editor_placeholders,
        save_file_button
    ]

    # Wire up the dropdown change event
    use_case_dropdown.change(
        fn=create_main_ui,
        inputs=use_case_dropdown,
        outputs=all_tab_outputs + [config_file_paths_state, train_params_keys_state, selected_tab_index_state]
    )

    # Wire up the save buttons (click events attached to placeholders)
    # Note: tab_index 0 is "Structured Train Config", so raw files start from index 1.
    save_file_button.click(
        fn=lambda uc, files, tab_idx, *contents:
            save_file_content(files[int(tab_idx) - 1], contents[int(tab_idx) - 1])
            if tab_idx > 0 and (int(tab_idx) - 1) < len(files) else "Cannot save structured config directly from this button.",
        inputs=[use_case_dropdown, config_file_paths_state, selected_tab_index_state] + file_editor_placeholders,
        outputs=output_log
    )
    save_train_button.click(
        fn=save_train_params,
        inputs=[use_case_dropdown, train_params_keys_state] + train_params_ui_placeholders,
        outputs=output_log
    )

    # Wire up other buttons
    train_button.click(fn=train, inputs=use_case_dropdown, outputs=output_log)
    refresh_button.click(
        fn=lambda: (gr.update(choices=get_use_cases(), value=get_use_cases()[0] if get_use_cases() else None)),
        inputs=[], # Changed from None to []
        outputs=use_case_dropdown
    ).then(
        fn=create_main_ui,
        inputs=use_case_dropdown,
        outputs=all_tab_outputs + [config_file_paths_state, train_params_keys_state, selected_tab_index_state]
    )

    # New: Add a handler to update selected_tab_index_state when a tab is selected
    main_tabs.select(
        fn=lambda evt: evt.index, # evt.index is the 0-indexed selected tab position
        inputs=[], # Changed from None to []
        outputs=selected_tab_index_state
    )

    # Initial load of the UI
    demo.load(
        fn=create_main_ui,
        inputs=gr.State(selected_use_case), # Pass the initial selected_use_case as a state
        outputs=all_tab_outputs + [config_file_paths_state, train_params_keys_state, selected_tab_index_state]
    )

if __name__ == "__main__":
    demo.launch(share=True)
