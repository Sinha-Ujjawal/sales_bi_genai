import importlib
import inspect
from typing import Any

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf


def _import_target(target: str):
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _filter_cfg(cfg: dict[str, Any]) -> dict:
    """Return a dict with only valid keys for the target class's __init__."""
    if "_target_" not in cfg:
        return cfg  # nothing to do for non-instantiable dicts

    target = cfg["_target_"]
    cls = _import_target(target)

    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # if constructor accepts **kwargs, pass all
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cfg

    valid_keys = {
        name
        for name, p in params.items()
        if name != "self"
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    return {k: v for k, v in cfg.items() if k == "_target_" or k in valid_keys}


def instantiate_filtered(cfg: Any, *args, **kwargs):
    """
    Wrapper around hydra.utils.instantiate that ignores extra keys
    not accepted by the target class's __init__.
    Works recursively.
    """
    # Convert to container so we can walk it
    container = OmegaConf.to_container(cfg, resolve=False)

    def recurse(obj):
        if isinstance(obj, dict) and "_target_" in obj:
            filtered = _filter_cfg(obj)
            # Recurse into values (nested configs)
            return {k: recurse(v) for k, v in filtered.items()}
        elif isinstance(obj, dict):
            return {k: recurse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recurse(v) for v in obj]
        else:
            return obj

    cleaned = recurse(container)

    # Re-wrap into DictConfig so Hydra instantiate can still resolve/interpolate
    cleaned_cfg = OmegaConf.create(cleaned)

    return hydra_instantiate(cleaned_cfg, *args, **kwargs)
