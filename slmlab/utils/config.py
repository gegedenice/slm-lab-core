from pathlib import Path
from types import SimpleNamespace
import yaml

def _to_ns(d: dict):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k,v in d.items()})
    if isinstance(d, list):
        return [_to_ns(x) for x in d]
    return d

def load_yaml(path: str|Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(use_case: str):
    cfg_path = Path(f"use_cases/{use_case}/configs/default.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file for use case '{use_case}' not found at {cfg_path}")

    cfg = load_yaml(cfg_path)

    # Resolve nested configs relative to the main config file's directory
    cfg_dir = cfg_path.parent
    for k in ["model", "method"]:
        p = cfg.get(k)
        if isinstance(p, str):
            nested_path = cfg_dir / p
            if nested_path.exists():
                cfg[k] = load_yaml(nested_path)

    return _to_ns(cfg)
