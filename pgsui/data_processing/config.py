from __future__ import annotations

import copy
import dataclasses
import logging
import os
import typing as t
from dataclasses import MISSING, asdict, fields
from dataclasses import is_dataclass
from dataclasses import is_dataclass as _is_dc
from typing import Any, Callable, Dict, Literal, Type, TypeVar

import yaml

# type variable for dataclass types
T = t.TypeVar("T")
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


def _is_dataclass_type(tp: t.Any) -> bool:
    """Return True if tp is a dataclass type (not instance).

    Args:
        tp (t.Any): A type to check.

    Returns:
        bool: True if `tp` is a dataclass type, False otherwise.
    """
    try:
        return isinstance(tp, type) and dataclasses.is_dataclass(tp)
    except Exception:
        return False


def _unwrap_optional(tp: t.Any) -> t.Any:
    """If Optional[T] or Union[T, None], return T; else tp."""
    origin = t.get_origin(tp)
    if origin is t.Union:
        args = [a for a in t.get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else tp
    return tp


def _expected_field_type(dc_type: type, name: str) -> t.Any:
    """Fetch the annotated type of field `name` on dataclass type `dc_type`.

    Args:
        dc_type (type): A dataclass type.
        name (str): The field name to look up.

    Returns:
        t.Any: The annotated type of the field.
    """
    for f in fields(dc_type):
        if f.name == name:
            return f.type
    raise KeyError(f"Unknown config key: '{name}' on {dc_type.__name__}")


def _instantiate_field(dc_type: type, name: str):
    """Create a default instance for nested dataclass field `name`.

    Attempts to use default_factory, then default, then type constructor.

    Args:
        dc_type (type): A dataclass type.
        name (str): The field name to instantiate.

    Returns:
        Any: An instance of the field's type.
    """
    for f in fields(dc_type):
        if f.name == name:
            # Prefer default_factory → default → type()
            if f.default_factory is not MISSING:  # type: ignore[attr-defined]
                return f.default_factory()
            if f.default is not MISSING:
                val = f.default
                # If default is None but type is dataclass, construct it:
                tp = _unwrap_optional(f.type)
                if val is None and _is_dataclass_type(tp):
                    return tp()
                return val
            # No default supplied; if it's a dataclass type, construct it.
            tp = _unwrap_optional(f.type)
            if _is_dataclass_type(tp):
                return tp()
            # Otherwise we cannot guess safely:
            raise KeyError(
                f"Cannot create default for '{name}' on {dc_type.__name__}; "
                "no default/default_factory and not a dataclass field."
            )
    raise KeyError(f"Unknown config key: '{name}' on {dc_type.__name__}'")


def _merge_mapping_into_dataclass(
    instance: T, payload: dict, *, path: str = "<root>"
) -> T:
    """Recursively merge a dict into a dataclass instance (strict on keys).

    Args:
        instance (T): A dataclass instance to update.
        payload (dict): A nested mapping to merge into `instance`.
        path (str): Internal use only; tracks the current path for error messages.

    Returns:
        T: The updated dataclass instance (same as `instance`).
    """
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass at {path}, got {type(instance)}")

    dc_type = type(instance)
    for k, v in payload.items():
        # Ensure field exists
        exp_type = _expected_field_type(dc_type, k)
        exp_core = _unwrap_optional(exp_type)

        cur = getattr(instance, k, MISSING)
        if cur is MISSING:
            raise KeyError(f"Unknown config key: '{path}.{k}'")

        if _is_dataclass_type(exp_core) and isinstance(v, dict):
            # Ensure we have a dataclass instance to merge into
            if cur is None or not is_dataclass(cur):
                cur = _instantiate_field(dc_type, k)
                setattr(instance, k, cur)
            merged = _merge_mapping_into_dataclass(cur, v, path=f"{path}.{k}")
            setattr(instance, k, merged)
        else:
            setattr(instance, k, _coerce_value(v, exp_core, f"{path}.{k}"))
    return instance


def _coerce_value(value: t.Any, tp: t.Any, where: str):
    """Lightweight coercion for common primitives and Literals.

    Args:
        value (t.Any): The input value to coerce.
        tp (t.Any): The target type annotation.
        where (str): Context string for error messages.

    Returns:
        t.Any: The coerced value, or the original if no coercion was applied.
    """
    origin = t.get_origin(tp)
    args = t.get_args(tp)

    # Literal[...] → restrict values
    if origin is t.Literal:
        allowed = set(args)
        if value not in allowed:
            raise ValueError(
                f"Invalid value for {where}. Expected one of {sorted(allowed)}, got {value!r}."
            )
        return value

    # Basic primitives coercion
    if tp in (int, float, bool, str):
        try:
            # Avoid bool('0') → True; handle common string bools
            if tp is bool and isinstance(value, str):
                v = value.strip().lower()
                if v in {"true", "1", "yes", "on"}:
                    return True
                if v in {"false", "0", "no", "off"}:
                    return False
                raise ValueError
            return tp(value)
        except Exception:
            # Fall through to return as-is; your code may purposely pass arrays, etc.
            return value

    # Dataclasses or other complex types → trust caller
    return value


def apply_dot_overrides(
    dc: t.Any,
    overrides: dict[str, t.Any] | None,
    *,
    root_cls: type | None = None,
    create_missing: bool = False,
    registry: dict[str, type] | None = None,
) -> t.Any:
    """Apply overrides like {'io.prefix': '...', 'train.batch_size': 64} to any \*Config dataclass.

    Args:
        dc (t.Any): A dataclass instance (or a dict that can be up-cast).
        overrides (dict[str, t.Any] | None): Mapping of dot-key paths to values.
        root_cls (type | None): Optional dataclass type to up-cast a root dict into (if `dc` is a dict).
        create_missing (bool): If True, instantiate missing intermediate dataclass nodes when the schema defines them.
        registry (dict[str, type] | None): Optional mapping from top-level segment → dataclass type to assist up-casting.

    Returns:
        t.Any: The updated dataclass instance (same object identity is not guaranteed; a deep copy is made).

    Notes:
        - No hard-coding of NLPCAConfig. Pass `root_cls=NLPCAConfig` (or UBPConfig, etc.) when starting from a dict.
        - Dict payloads encountered at intermediate nodes are merged into the expected dataclass type using schema introspection.
        - Enforces unknown-key errors to keep configs honest.
    """
    if not overrides:
        return dc

    # Root up-cast if needed
    if not is_dataclass(dc):
        if isinstance(dc, dict):
            if root_cls is None:
                raise TypeError(
                    "Root payload is a dict. Provide `root_cls` to up-cast it into the desired *Config dataclass."
                )
            base = root_cls()
            dc = _merge_mapping_into_dataclass(base, dc)
        else:
            raise TypeError(
                "apply_dot_overrides expects a dataclass instance or a dict for up-cast."
            )

    updated = copy.deepcopy(dc)

    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        node = updated
        node_type = type(node)

        # Descend to parent
        for idx, seg in enumerate(parts[:-1]):
            if not is_dataclass(node):
                parent_path = ".".join(parts[:idx]) or "<root>"
                raise KeyError(
                    f"Target '{parent_path}' is not a dataclass in the override path; cannot descend into non-dataclass objects."
                )

            # Validate field existence and fetch expected type
            exp_type = _expected_field_type(node_type, seg)
            exp_core = _unwrap_optional(exp_type)

            # Materialize or up-cast if needed
            child = getattr(node, seg, MISSING)
            if child is MISSING:
                raise KeyError(f"Unknown config key: '{'.'.join(parts[:idx+1])}'")

            if isinstance(child, dict) and _is_dataclass_type(exp_core):
                # Up-cast dict → dataclass of the expected type
                child = _merge_mapping_into_dataclass(
                    exp_core(), child, path=".".join(parts[: idx + 1])
                )
                setattr(node, seg, child)

            if child is None and create_missing and _is_dataclass_type(exp_core):
                child = exp_core()
                setattr(node, seg, child)

            node = getattr(node, seg)
            node_type = type(node)

        # Assign leaf with light coercion
        if not is_dataclass(node):
            parent_path = ".".join(parts[:-1]) or "<root>"
            raise KeyError(
                f"Target '{parent_path}' is not a dataclass in the override path; cannot set '{parts[-1]}'."
            )

        leaf = parts[-1]

        # Check field exists and coerce to its annotated type
        exp_type = _expected_field_type(type(node), leaf)
        exp_core = _unwrap_optional(exp_type)

        if not hasattr(node, leaf):
            raise KeyError(f"Unknown config key: '{dotkey}'")

        setattr(node, leaf, _coerce_value(value, exp_core, dotkey))

    return updated
