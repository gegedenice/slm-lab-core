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

def load_config(cfg_path: str|Path):
    cfg = load_yaml(cfg_path)
    for k in ["model","method"]:
        p = cfg.get(k)
        if isinstance(p, str) and Path(p).exists():
            cfg[k] = load_yaml(p)
    return _to_ns(cfg)
