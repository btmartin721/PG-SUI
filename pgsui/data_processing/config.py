from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Literal, Type, TypeVar

import yaml

# type variable for dataclass types
T = TypeVar("T")

"""
Config utilities for PG-SUI.

We keep nested configs as dataclasses at all times.

Public API:
- load_yaml_to_dataclass
- apply_dot_overrides
- dataclass_to_yaml
- save_dataclass_yaml
"""


# ---------------- Env var interpolation ----------------
def _interpolate_env(s: str) -> str:
    """Interpolate env vars in a string.

    Syntax: ${VAR} or ${VAR:default}

    Args:
        s (str): Input string possibly containing env var patterns.

    Returns:
        str: The string with env vars interpolated.
    """
    out, i = [], 0
    while i < len(s):
        if s[i : i + 2] == "${":
            j = s.find("}", i + 2)
            if j == -1:
                out.append(s[i:])
                break
            token = s[i + 2 : j]
            if ":" in token:
                var, default = token.split(":", 1)
                out.append(os.getenv(var, default))
            else:
                out.append(os.getenv(token, ""))
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _walk_env(obj: Any) -> Any:
    """Recursively interpolate env vars in strings within a nested structure.

    Args:
        obj (Any): The input object, which can be a string, dict, list, or other types.

    Returns:
        Any: The object with environment variables interpolated in strings.
    """
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_env(v) for v in obj]
    return obj


# ---------------- YAML helpers ----------------
def dataclass_to_yaml(dc: T) -> str:
    """Convert a dataclass instance to a YAML string.

    Args:
        dc (T): A dataclass instance.

    Returns:
        str: The YAML representation of the dataclass.
    """
    if not is_dataclass(dc):
        raise TypeError("dataclass_to_yaml expects a dataclass instance.")
    return yaml.safe_dump(asdict(dc), sort_keys=False)


def save_dataclass_yaml(dc: T, path: str) -> None:
    """Save a dataclass instance as a YAML file.

    Args:
        dc (T): A dataclass instance.
        path (str): Path to save the YAML file.

    Raises:
        TypeError: If `dc` is not a dataclass instance.
    """
    if not is_dataclass(dc):
        raise TypeError(
            "save_dataclass_yaml expects a dataclass or dataclass instance."
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(dataclass_to_yaml(dc))


def _pop_preset(raw: Dict[str, Any]) -> str | None:
    """Extract and remove 'preset' from raw dict if present.

    Args:
        raw (Dict[str, Any]): The raw dictionary to inspect.

    Returns:
        str | None: The preset value if found, otherwise None.
    """
    if isinstance(raw.get("preset"), str):
        return raw.pop("preset")
    io = raw.get("io")
    if isinstance(io, dict) and isinstance(io.get("preset"), str):
        return io.pop("preset")
    return None


def _merge_into_dataclass(inst: Any, payload: Dict[str, Any], path: str = "") -> Any:
    """Recursively merge a nested dict into a dataclass instance in place.

    Args:
        inst (Any): A dataclass instance to update.
        payload (Dict[str, Any]): A nested mapping to merge into `inst`.
        path (str): Internal use only; tracks the current path for error messages.

    Returns:
        Any: The updated dataclass instance (same as `inst`).

    Raises:
        TypeError: If `inst` is not a dataclass.
        KeyError: If `payload` contains keys not present in `inst`.
    """
    from dataclasses import fields
    from dataclasses import is_dataclass as _is_dc

    if not _is_dc(inst):
        raise TypeError(
            f"_merge_into_dataclass expects a dataclass at '{path or '<root>'}'"
        )

    fld_map = {f.name: f for f in fields(inst)}
    for k, v in payload.items():
        if k not in fld_map:
            full = f"{path + '.' if path else ''}{k}"
            raise KeyError(f"Unknown key '{full}'")
        cur = getattr(inst, k)
        if _is_dc(cur) and isinstance(v, dict):
            _merge_into_dataclass(cur, v, path=(f"{path}.{k}" if path else k))
        else:
            setattr(inst, k, v)
    return inst


def load_yaml_to_dataclass(
    path: str,
    dc_type: Type[T],
    *,
    base: T | None = None,
    overlays: Dict[str, Any] | None = None,
    preset_builder: Callable[[str], T] | None = None,
    yaml_preset_behavior: Literal["ignore", "error"] = "ignore",
) -> T:
    """Load a YAML file and merge into a dataclass instance with strict precedence.

    This function is designed for the new argument hierarchy: defaults < CLI preset (build `base` from it) < YAML file < CLI args/--set

    Notes:
        - `preset` is **CLI-only**. If the YAML contains `preset`, it will be ignored (default) or cause an error depending on `yaml_preset_behavior`.
        - Pass a `base` instance that is already constructed from the CLI-selected preset (e.g., `NLPCAConfig.from_preset(args.preset)`), and this function will overlay the YAML on top of it. Any additional `overlays` (a nested dict) are applied last.

    Args:
        path (str): Path to the YAML file.
        dc_type (Type[T]): Dataclass type to construct if `base` is not provided.
        base (T | None): A preconstructed dataclass instance to start from
            (typically built from the CLI preset). If provided, it takes precedence
            over any other starting point.
        overlays (Dict[str, Any] | None): A nested mapping to apply **after** the
            YAML (e.g., derived CLI flags). These win over YAML values.
        preset_builder (Callable[[str], T] | None): Retained for backward
            compatibility. Not used when enforcing CLI-only presets.
        yaml_preset_behavior (Literal["ignore","error"]): What to do if the YAML
            contains a `preset` key. Default: "ignore".

    Returns:
        T: The merged dataclass instance.

    Raises:
        TypeError: If `base` is not a dataclass, or YAML root isn't a mapping,
            or `overlays` isn't a mapping when provided.
        ValueError: If `yaml_preset_behavior="error"` and YAML contains `preset`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _walk_env(raw)

    # Enforce: 'preset' is CLI-only.
    if isinstance(raw, dict) and "preset" in raw:
        preset_in_yaml = raw.get("preset")
        if yaml_preset_behavior == "error":
            raise ValueError(
                f"YAML contains 'preset: {preset_in_yaml}'. "
                "The preset must be selected via the command line only."
            )
        # ignore (default): drop it and continue
        logging.warning(
            "Ignoring 'preset' in YAML (%r). Preset selection is CLI-only.",
            preset_in_yaml,
        )
        raw.pop("preset", None)

    # Start from `base` if given; else construct a fresh instance of dc_type.
    if base is not None:
        if not is_dataclass(base):
            raise TypeError("`base` must be a dataclass instance.")
        cfg = copy.deepcopy(base)
    else:
        # Do NOT call preset_builder here; presets are CLI-only.
        cfg = dc_type()  # defaults

    if not isinstance(raw, dict):
        raise TypeError(f"{path} did not parse as a mapping.")

    # YAML overlays the starting config
    _merge_into_dataclass(cfg, raw)

    # Optional final overlays (e.g., mapped CLI flags / --set already parsed)
    if overlays:
        if not isinstance(overlays, dict):
            raise TypeError("`overlays` must be a nested dict.")
        _merge_into_dataclass(cfg, overlays)

    return cfg


# ---------------- Dot-key overrides (with ROOT self-heal) ----------------
def apply_dot_overrides(dc: Any, overrides: Dict[str, Any] | None) -> Any:
    """Apply overrides like {"io.prefix": "...", "train.batch_size": 64}.

    Args:
        dc (Any): A dataclass instance to update.
        overrides (Dict[str, Any] | None): A mapping of dot-key paths to values

    Returns:
        Any: The updated dataclass instance (same as `dc`).

    Notes:
        - If `dc` is accidentally a dict (not a dataclass), we up-cast it into an NLPCAConfig instance and continue (so callers survive older code paths).
    """
    if not overrides:
        return dc

    # Root self-heal: dict -> NLPCAConfig
    if not is_dataclass(dc):
        if isinstance(dc, dict):
            try:
                # Local import to avoid cycles for general use
                from pgsui.data_processing.containers import NLPCAConfig

                base = NLPCAConfig()
                dc = _merge_into_dataclass(base, dc)
            except Exception as e:
                raise TypeError(
                    "apply_dot_overrides expects a dataclass instance as `dc` "
                    "and failed to up-cast from dict payload."
                ) from e
        else:
            raise TypeError("apply_dot_overrides expects a dataclass instance as `dc`.")

    updated = copy.deepcopy(dc)

    for dotkey, value in (overrides or {}).items():
        parts = dotkey.split(".")
        node = updated

        # descend to parent node
        for idx, seg in enumerate(parts[:-1]):
            if not is_dataclass(node):
                parent_path = ".".join(parts[:idx]) or "<root>"
                raise KeyError(
                    f"Target '{parent_path}' is not a dataclass in the override path; "
                    "cannot descend into non-dataclass objects."
                )
            if not hasattr(node, seg):
                raise KeyError(f"Unknown config key: '{'.'.join(parts[:idx+1])}'")

            child = getattr(node, seg)

            # If child is dict but the field is structured, try to up-cast using a blank sibling
            if isinstance(child, dict):
                try:
                    blank = type(node)()  # new instance of this dataclass
                    blank_child = getattr(blank, seg)
                    if is_dataclass(blank_child):
                        child = _merge_into_dataclass(
                            blank_child, child, path=".".join(parts[: idx + 1])
                        )
                        setattr(node, seg, child)
                except Exception:
                    pass  # fall through

            node = getattr(node, seg)

        # assign leaf
        if not is_dataclass(node):
            parent_path = ".".join(parts[:-1]) or "<root>"
            raise KeyError(
                f"Target '{parent_path}' is not a dataclass in the override path; cannot set '{parts[-1]}'."
            )
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise KeyError(f"Unknown config key: '{dotkey}'")
        setattr(node, leaf, value)

    return updated
