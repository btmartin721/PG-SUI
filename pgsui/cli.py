#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PG-SUI Imputation CLI

Argument-precedence model:
    code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set k=vx

Notes
-----
- Preset is a CLI-only choice and will be respected unless overridden by YAML or CLI.
- YAML entries override preset (a 'preset' key in YAML is ignored with a warning).
- CLI flags only override when explicitly provided (argparse uses SUPPRESS).
- --set key=value has the highest precedence and applies dot-path overrides.

Examples
--------
pg-sui --vcf data.vcf.gz --popmap pops.popmap --prefix run1
pg-sui --vcf data.vcf.gz --popmap pops.popmap --prefix tuned --tune
pg-sui --vcf data.vcf.gz --popmap pops.popmap --prefix demo \
    --models ImputeAutoencoder ImputeVAE ImputeUBP --seed deterministic --verbose
pg-sui --vcf data.vcf.gz --popmap pops.popmap --prefix subset \
    --include-pops EA GU TT ON --device cpu --sim-prop 0.3 --sim-strategy nonrandom
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    cast,
)

from snpio import (
    GenePopReader,
    PhylipReader,
    SNPioMultiQC,
    StructureReader,
    TreeParser,
    VCFReader,
)

from pgsui import (
    AutoencoderConfig,
    ImputeAutoencoder,
    ImputeMostFrequent,
    ImputeNLPCA,
    ImputeUBP,
    ImputeRefAllele,
    ImputeVAE,
    MostFrequentConfig,
    RefAlleleConfig,
    VAEConfig,
    NLPCAConfig,
    UBPConfig,
)
from pgsui.data_processing.config import (
    apply_dot_overrides,
    dataclass_to_yaml,
    load_yaml_to_dataclass,
    save_dataclass_yaml,
)
from pgsui.data_processing.containers import NLPCAConfig, UBPConfig

# Canonical model order used everywhere (default and subset ordering)
MODEL_ORDER: Tuple[str, ...] = (
    "ImputeUBP",
    "ImputeNLPCA",
    "ImputeVAE",
    "ImputeAutoencoder",
    "ImputeMostFrequent",
    "ImputeRefAllele",
)

# Strategies supported by SimMissingTransformer + SimConfig.
SIM_STRATEGY_CHOICES: Tuple[str, ...] = (
    "random",
    "random_weighted",
    "random_weighted_inv",
    "nonrandom",
    "nonrandom_weighted",
)

P = ParamSpec("P")
R = TypeVar("R")


# ----------------------------- CLI Utilities ----------------------------- #
def _print_version() -> None:
    """Print PG-SUI version and exit."""
    from pgsui import __version__ as version

    logging.info(f"Using PG-SUI version: {version}")


def _model_family(model_name: str) -> str:
    """Return output family folder name used by PG-SUI."""
    if model_name in {"ImputeUBP", "ImputeNLPCA", "ImputeVAE", "ImputeAutoencoder"}:
        return "Unsupervised"
    if model_name in {"ImputeMostFrequent", "ImputeRefAllele"}:
        return "Deterministic"
    return "Unknown"


def _flatten_dict(d: dict, parent: str = "") -> dict:
    """Flatten a nested dict into dot keys."""
    out: dict = {}
    for k, v in (d or {}).items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _force_tuning_off(cfg: Any, model_name: str) -> Any:
    """Force tuning disabled on a config object (best-effort, but strict for tune-capable models)."""
    # Prefer direct attribute mutation (avoids apply_dot_overrides edge-cases)
    try:
        if hasattr(cfg, "tune") and hasattr(cfg.tune, "enabled"):
            cfg.tune.enabled = False
            return cfg
    except Exception:
        pass

    # Fallback to dot override
    try:
        return apply_dot_overrides(cfg, {"tune.enabled": False})
    except Exception as e:
        # Only strict for models that actually support tuning
        if model_name in {"ImputeUBP", "ImputeNLPCA", "ImputeVAE", "ImputeAutoencoder"}:
            raise RuntimeError(
                f"Failed to force tuning off for {model_name}: {e}"
            ) from e
        return cfg


def _find_best_params_json(prefix: str, model_name: str) -> Path | None:
    """Locate best parameter JSON (tuned or final) for a model.

    Args:
        prefix (str): Output prefix used during the run.
        model_name (str): Model name to look for.

    Returns:
        Path | None: Path to best_parameters.json / best_tuned_parameters.json if found; else None.
    """
    families = ("Unsupervised", "Deterministic")
    model_dir_candidates = (model_name, model_name.lower())

    for fam in families:
        for mdir in model_dir_candidates:
            base = Path(f"{prefix}_output") / fam
            candidates = (
                base / "optimize" / mdir / "parameters" / "best_tuned_parameters.json",
                base / "parameters" / mdir / "best_parameters.json",
            )
            for p in candidates:
                if p.exists():
                    return p
    return None


def _load_best_params(best_params_path: Path) -> dict:
    """Load best parameters JSON."""
    with best_params_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"best_parameters.json must be a JSON object, got {type(data)}"
        )
    return data


def _apply_best_params_to_cfg(cfg: Any, best_params: dict, model_name: str) -> Any:
    """Apply best params into cfg using dot-path keys or inferred dot-paths.

    Remaps legacy JSON keys to Config attribute names (e.g. power -> weights_power). Remaps 'model_params' dict to 'model' dot-paths. Ignores known metadata keys (n_features, num_classes) to prevent warning spam. Applies keys via prefixes (model., train., etc.) if explicit paths fail.
    """
    # 1. Map JSON keys to Config attribute names
    #    (JSON Key -> Dataclass Attribute Name)
    key_aliases = {
        "power": "weights_power",
        "inverse": "weights_inverse",
        "normalize": "weights_normalize",
    }

    # 2. Define keys to explicitly ignore (metadata in JSON that isn't a config setting)
    #    This prevents "Best param 'n_features' not recognized" warnings.
    ignored_keys = {
        "n_features",
        "num_classes",
        "hidden_layer_sizes",  # This is derived from layer_schedule/scaling_factor
        "model_params.n_features",
        "model_params.num_classes",
        "model_params.hidden_layer_sizes",
    }

    # 3. Flatten dictionary with intelligence
    flat = {}
    for k, v in best_params.items():
        # A. Handle the specific 'model_params' block
        if str(k) == "model_params" and isinstance(v, dict):
            # Map "model_params.x" -> "model.x"
            # We force the parent to be "model" so it targets ModelConfig
            nested = _flatten_dict(v, parent="model")
            flat.update(nested)
            continue

        # B. Standard Flattening
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, str(k)))
        else:
            # C. Apply Aliases for root keys
            final_k = str(k)
            if final_k in key_aliases:
                final_k = key_aliases[final_k]
            flat[final_k] = v

    # 4. Apply settings
    #    Note: 'tune.' is usually not needed for best_params (which are the result of tuning),
    #    but kept for safety.
    candidate_prefixes = ("", "model.", "train.", "sim.", "tune.", "io.", "plot.")

    for raw_k, v in flat.items():
        # Skip explicitly ignored metadata
        if raw_k in ignored_keys or any(
            raw_k.endswith(f".{ik}") for ik in ignored_keys
        ):
            continue

        # CASE 1: Key already has dots (e.g. "model.latent_dim")
        # Because we remapped model_params -> model above, these should now be valid.
        if "." in raw_k:
            try:
                cfg = apply_dot_overrides(cfg, {raw_k: v})
                continue
            except Exception:
                # If exact dot-path failed, fall through to prefix search
                pass

        # CASE 2: Prefix Search
        # Try applying empty prefix, then model., train., etc.
        applied = False
        for pref in candidate_prefixes:
            k = f"{pref}{raw_k}" if pref else raw_k
            try:
                cfg = apply_dot_overrides(cfg, {k: v})
                applied = True
                break
            except Exception:
                continue

        if not applied:
            logging.warning(
                f"Best param '{raw_k}' not recognized for {model_name}; leaving config unchanged for that key."
            )

    return cfg


def _configure_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    """Configure root logger.

    Args:
        verbose (bool): If True, INFO; else ERROR.
        log_file (Optional[str]): Optional file to tee logs to.
    """
    level = logging.INFO if verbose else logging.ERROR
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _parse_seed(seed_arg: str) -> Optional[int]:
    """Parse --seed argument into an int or None."""
    s = seed_arg.strip().lower()
    if s == "random":
        return None
    if s == "deterministic":
        return 42
    try:
        return int(seed_arg)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Invalid --seed. Use 'random', 'deterministic', or an integer."
        ) from e


def _parse_models(models: Iterable[str]) -> Tuple[str, ...]:
    """Validate and canonicalize model names in a deterministic order.

    - If no models are provided, returns all in MODEL_ORDER.
    - If a subset is provided via --models, returns them in MODEL_ORDER order.
    """
    models = tuple(models)  # in case it's a generator
    valid = set(MODEL_ORDER)

    # Validate first
    unknown = [m for m in models if m not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown model(s): {unknown}. Valid options: {list(MODEL_ORDER)}"
        )

    # Default: all models in canonical order
    if not models:
        return MODEL_ORDER

    # Subset: keep only those requested, but in canonical order
    selected = tuple(m for m in MODEL_ORDER if m in models)
    return selected


def _parse_overrides(pairs: list[str]) -> dict:
    """Parse --set key=value into typed values via literal_eval."""
    out: dict = {}
    for kv in pairs or []:
        if "=" not in kv:
            raise argparse.ArgumentTypeError(f"--set expects key=value, got '{kv}'")
        k, v = kv.split("=", 1)
        v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v  # raw string fallback
    return out


def _parse_allele_encoding(arg: str) -> dict:
    """Parse STRUCTURE allele encoding dict from JSON or Python literal."""
    try:
        payload = json.loads(arg)
    except Exception:
        try:
            payload = ast.literal_eval(arg)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Invalid --structure-allele-encoding; must be a dict. Error: {e}"
            ) from e

    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError(
            "--structure-allele-encoding must be a dict-like mapping."
        )

    out: dict = {}
    for k, v in payload.items():
        key = k
        if isinstance(k, str):
            k_strip = k.strip()
            if k_strip.lstrip("-").isdigit():
                try:
                    key = int(k_strip)
                except Exception:
                    key = k
        out[key] = str(v)
    return out


def _normalize_input_format(fmt: str) -> str:
    """Normalize format aliases into canonical reader names."""
    fmt = fmt.lower()
    if fmt in {"vcf", "vcf.gz"}:
        return "vcf"
    if fmt in {"phy", "phylip"}:
        return "phylip"
    if fmt in {"gen", "genepop"}:
        return "genepop"
    if fmt in {"str", "structure"}:
        return "structure"
    return fmt


def _normalize_plot_format(fmt: str) -> Literal["pdf", "png", "jpg", "svg"]:
    """Normalize plot format aliases to reader-supported values."""
    fmt = fmt.lower()
    if fmt == "jpeg":
        return "jpg"
    return cast(Literal["pdf", "png", "jpg", "svg"], fmt)


def _expand_path(path: str | None) -> str | None:
    """Expand ~ and env vars in a path-like string."""
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    expanded = os.path.expandvars(raw)
    return str(Path(expanded).expanduser())


def _resolve_tree_paths(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None]:
    """Resolve tree-related paths from CLI args."""
    treefile = _expand_path(getattr(args, "treefile", None))
    qmatrix = _expand_path(getattr(args, "qmatrix", None))
    siterates = _expand_path(getattr(args, "siterates", None))
    return treefile, qmatrix, siterates


def _config_needs_tree(cfg: Any | None) -> bool:
    """Return True if config requires a tree parser for simulated missingness."""
    if cfg is None:
        return False
    sim_cfg = getattr(cfg, "sim", None)
    if sim_cfg is None:
        return False
    strategy = getattr(sim_cfg, "sim_strategy", None)
    simulate = bool(getattr(sim_cfg, "simulate_missing", False))
    return (
        simulate
        and isinstance(strategy, str)
        and strategy in {"nonrandom", "nonrandom_weighted"}
    )


def _args_to_cli_overrides(args: argparse.Namespace) -> dict:
    """Convert explicitly provided CLI flags into config dot-overrides."""
    overrides: dict = {}

    # IO / top-level controls
    if hasattr(args, "prefix") and args.prefix is not None:
        overrides["io.prefix"] = args.prefix
    else:
        # Prefer --input stem; fallback to legacy --vcf stem
        input_path = getattr(args, "input", None)
        if input_path is None and hasattr(args, "vcf"):
            input_path = getattr(args, "vcf", None)
        if input_path:
            overrides["io.prefix"] = str(Path(input_path).stem)

    if hasattr(args, "verbose"):
        overrides["io.verbose"] = bool(args.verbose)
    if hasattr(args, "n_jobs"):
        overrides["io.n_jobs"] = int(args.n_jobs)
    if hasattr(args, "seed"):
        overrides["io.seed"] = _parse_seed(args.seed)
    if hasattr(args, "debug"):
        overrides["io.debug"] = bool(args.debug)

    # Train
    if hasattr(args, "batch_size"):
        overrides["train.batch_size"] = int(args.batch_size)
    if hasattr(args, "device"):
        dev = args.device
        if dev == "cuda":
            dev = "gpu"
        overrides["train.device"] = dev

    # Plot
    if hasattr(args, "plot_format"):
        overrides["plot.fmt"] = args.plot_format
    if getattr(args, "disable_plotting", False):
        logging.info(
            "Disabling plotting for all models as per --disable-plotting flag."
        )
        overrides["plot.show"] = False

    # Simulation overrides
    if hasattr(args, "sim_strategy"):
        overrides["sim.sim_strategy"] = args.sim_strategy
    if hasattr(args, "sim_prop"):
        overrides["sim.sim_prop"] = float(args.sim_prop)

    # Tuning
    if hasattr(args, "tune_metrics"):
        overrides["tune.metrics"] = args.tune_metrics

    if getattr(args, "load_best_params", False):
        # Never allow CLI flags to re-enable tuning when loading params
        if hasattr(args, "tune") and bool(getattr(args, "tune", False)):
            logging.warning(
                "--tune was supplied, but --load-best-params is active; ignoring --tune."
            )
        if hasattr(args, "tune_n_trials"):
            logging.warning(
                "--tune-n-trials was supplied, but --load-best-params is active; ignoring it."
            )
    else:
        if hasattr(args, "tune"):
            overrides["tune.enabled"] = bool(args.tune)
        if hasattr(args, "tune_n_trials"):
            overrides["tune.n_trials"] = int(args.tune_n_trials)

    return overrides


def _format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def log_model_time(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator to time run_model_safely; assumes model_name is first arg."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        model_name = str(args[0]) if args else "<unknown model>"
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            elapsed = time.perf_counter() - start
            logging.error(
                f"{model_name} failed after {elapsed:0.2f}s "
                f"({_format_seconds(elapsed)}).",
                exc_info=True,
            )
            raise
        elapsed = time.perf_counter() - start
        logging.info(
            f"{model_name} finished in {elapsed:0.2f}s "
            f"({_format_seconds(elapsed)})."
        )
        return result

    return cast(Callable[P, R], wrapper)


# ------------------------------ Core Runner ------------------------------ #
def build_genotype_data(
    input_path: str,
    fmt: Literal[
        "vcf",
        "vcf.gz",
        "phy",
        "phylip",
        "genepop",
        "gen",
        "structure",
        "str",
    ],
    popmap_path: str | None,
    treefile: str | None,
    qmatrix: str | None,
    siterates: str | None,
    force_popmap: bool,
    debug: bool,
    include_pops: List[str] | None,
    plot_format: Literal["pdf", "png", "jpg", "jpeg", "svg"],
    structure_has_popids: bool = False,
    structure_has_marker_names: bool = False,
    structure_allele_start_col: int | None = None,
    structure_allele_encoding: dict | None = None,
):
    """Load genotype data from heterogeneous inputs.

    Args:
        input_path (str): Path to genotype data file.
        fmt (Literal): Format of genotype data file.
        popmap_path (str | None): Optional path to population map file.
        treefile (str | None): Optional path to phylogenetic tree file.
        qmatrix (str | None): Optional path to IQ-TREE Q matrix file.
        siterates (str | None): Optional path to SNP site rates file.
        force_popmap (bool): Whether to force use of popmap even if samples don't match exactly.
        debug (bool): Whether to enable debug-level logging in SNPio readers.
        include_pops (List[str] | None): Optional list of population IDs to include.
        plot_format (Literal): Figure format for SNPio plots.
        structure_has_popids (bool): STRUCTURE only; whether pop IDs are present.
        structure_has_marker_names (bool): STRUCTURE only; whether the first line has marker names.
        structure_allele_start_col (int | None): STRUCTURE only; zero-based allele start column.
        structure_allele_encoding (dict | None): STRUCTURE only; allele encoding map.
    """
    fmt_norm = _normalize_input_format(fmt)
    plot_format = _normalize_plot_format(cast(str, plot_format))
    logging.info(f"Loading {fmt_norm.upper()} and popmap data...")

    kwargs = {
        "filename": input_path,
        "popmapfile": popmap_path,
        "force_popmap": force_popmap,
        "verbose": debug,
        "include_pops": include_pops if include_pops else None,
        "prefix": f"snpio_{Path(input_path).stem}",
        "plot_format": plot_format,
    }

    if fmt_norm == "vcf":
        gd = VCFReader(**kwargs)
    elif fmt_norm == "phylip":
        gd = PhylipReader(**kwargs)
    elif fmt_norm == "genepop":
        gd = GenePopReader(**kwargs)
    elif fmt_norm == "structure":
        kwargs.update(
            {
                "has_popids": structure_has_popids,
                "has_marker_names": structure_has_marker_names,
                "allele_start_col": structure_allele_start_col,
                "allele_encoding": structure_allele_encoding,
            }
        )
        gd = StructureReader(**kwargs)
    else:
        raise ValueError(f"Unsupported genotype data format: {fmt}")

    tp = None
    if treefile is not None:
        logging.info("Parsing phylogenetic tree...")

        tp = TreeParser(
            gd, treefile=treefile, qmatrix=qmatrix, siterates=siterates, verbose=True
        )

    logging.info("Loaded genotype data.")
    return gd, tp


@log_model_time
def run_model_safely(model_name: str, builder, *, warn_only: bool = True) -> None:
    """Run model builder + fit/transform with error isolation."""
    logging.info(f"▶ Running {model_name} ...")
    try:
        model = builder()
        model.fit()
        X_imputed = model.transform()
        logging.info(f"✓ {model_name} completed.")
        return X_imputed
    except Exception as e:
        if warn_only:
            logging.warning(f"⚠ {model_name} failed: {e}", exc_info=True)
        else:
            raise


# -------------------------- Model Registry ------------------------------- #
# Add config-driven models here by listing the class and its config dataclass.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ImputeUBP": {"cls": ImputeUBP, "config_cls": UBPConfig},
    "ImputeNLPCA": {"cls": ImputeNLPCA, "config_cls": NLPCAConfig},
    "ImputeAutoencoder": {"cls": ImputeAutoencoder, "config_cls": AutoencoderConfig},
    "ImputeVAE": {"cls": ImputeVAE, "config_cls": VAEConfig},
    "ImputeMostFrequent": {"cls": ImputeMostFrequent, "config_cls": MostFrequentConfig},
    "ImputeRefAllele": {"cls": ImputeRefAllele, "config_cls": RefAlleleConfig},
}


def _build_effective_config_for_model(
    model_name: str, args: argparse.Namespace
) -> Any | None:
    """Build the effective config object for a specific model (if it has one).

    Precedence (lowest → highest):
        defaults < preset (--preset) < YAML (--config) < explicit CLI flags < --set

    Returns:
        Config dataclass instance or None (for models without config dataclasses).
    """
    reg = MODEL_REGISTRY[model_name]
    cfg_cls = reg.get("config_cls")

    if cfg_cls is None:
        return None

    # 0) Start from pure dataclass defaults.
    cfg = cfg_cls()

    # 1) If user explicitly provided a preset, overlay it.
    if hasattr(args, "preset"):
        preset_name = args.preset
        cfg = cfg_cls.from_preset(preset_name)
        logging.info(f"Initialized {model_name} from '{preset_name}' preset.")
    else:
        logging.info(f"Initialized {model_name} from dataclass defaults (no preset).")

    # 2) YAML overlays preset/defaults (boss). Ignore any 'preset' in YAML.
    yaml_path = getattr(args, "config", None)

    if yaml_path:
        cfg = load_yaml_to_dataclass(
            yaml_path,
            cfg_cls,
            base=cfg,
            yaml_preset_behavior="ignore",  # 'preset' key in YAML ignored with warning
        )
        logging.info(
            f"Loaded YAML config for {model_name} from {yaml_path} (ignored 'preset' in YAML if present)."
        )

    # 3) Optional: load best parameters from a previous run and force tuning OFF.
    if getattr(args, "load_best_params", False):
        # Determine which prefix to look under for *_output
        src_prefix = getattr(args, "best_params_prefix", None)
        if src_prefix is None:
            # Use the resolved prefix if provided; otherwise fall back to input
            # stem behavior
            src_prefix = getattr(args, "prefix", None)

            if src_prefix is None and hasattr(args, "vcf"):
                src_prefix = str(Path(args.vcf).stem)

            if src_prefix is None:
                # As a last resort, use current effective io.prefix if it exists in cfg
                src_prefix = getattr(getattr(cfg, "io", object()), "prefix", None)

        if getattr(args, "tune", False):
            logging.warning(
                "--tune was supplied, but --load-best-params is active; forcing tuning OFF."
            )

        # Force tuning disabled in config (even if CLI/YAML enabled it)
        cfg = _force_tuning_off(cfg, model_name)

        best_path = _find_best_params_json(str(src_prefix), model_name)
        if best_path is None:
            # For tune-capable (unsupervised) models, treat as an error; deterministic models warn only.
            fam = _model_family(model_name)
            msg = (
                "Requested --load-best-params, but could not find a best parameters JSON "
                f"for {model_name}. Looked under '.../optimize/<model>/parameters/best_tuned_parameters.json' and '{src_prefix}_output/{fam}/parameters/{model_name}/best_parameters.json'"
            )
            if model_name in {
                "ImputeUBP",
                "ImputeNLPCA",
                "ImputeVAE",
                "ImputeAutoencoder",
            }:
                logging.error(msg)
                raise FileNotFoundError(msg)
            logging.warning(msg)
        else:
            logging.info(f"Loading best parameters for {model_name} from: {best_path}")
            best_params = _load_best_params(best_path)
            cfg = _apply_best_params_to_cfg(cfg, best_params, model_name)
            cfg = _force_tuning_off(cfg, model_name)

    # 4) Explicit CLI flags overlay YAML/best-params layers.
    cli_overrides = _args_to_cli_overrides(args)
    if cli_overrides:
        cfg = apply_dot_overrides(cfg, cli_overrides)

    # Keep tuning disabled if --load-best-params was requested, even if CLI flags tried to re-enable it.
    if getattr(args, "load_best_params", False):
        cfg = _force_tuning_off(cfg, model_name)

    # 5) --set has highest precedence.
    user_overrides = _parse_overrides(getattr(args, "set", []))

    if user_overrides:
        try:
            cfg = apply_dot_overrides(cfg, user_overrides)
        except Exception as e:
            if model_name in {
                "ImputeUBP",
                "ImputeNLPCA",
                "ImputeAutoencoder",
                "ImputeVAE",
            }:
                logging.error(
                    f"Error applying --set overrides to {model_name} config: {e}"
                )
                raise
            else:
                pass  # non-config-driven models ignore --set

        # FINAL GUARANTEE:
        # --load-best-params always wins over
        # --set, YAML, preset, and CLI flags.
        if getattr(args, "load_best_params", False):
            # If user explicitly tried to set tune.* via --set, warn and override.
            if any(str(k).startswith("tune.") for k in (user_overrides or {}).keys()):
                logging.warning(
                    f"{model_name}: '--set tune.*=...' was provided, but --load-best-params forces tuning OFF. "
                    "Ignoring any tune.* overrides."
                )
            cfg = _force_tuning_off(cfg, model_name)

    return cfg


def _maybe_print_or_dump_configs(
    cfgs_by_model: Dict[str, Any], args: argparse.Namespace
) -> bool:
    """Handle --print-config / --dump-config for ALL config-driven models selected.

    Returns:
        True if we printed/dumped and should exit; else False.
    """
    did_io = False
    if getattr(args, "print_config", False):
        for m, cfg in cfgs_by_model.items():
            if cfg is None:
                continue
            print(f"# --- {m} effective config ---")
            print(dataclass_to_yaml(cfg))
            print()
        did_io = True

    if hasattr(args, "dump_config") and args.dump_config:
        # If multiple models, add suffix per model (before extension if possible)
        dump_base = args.dump_config
        for m, cfg in cfgs_by_model.items():
            if cfg is None:
                continue
            if "." in dump_base:
                stem, ext = dump_base.rsplit(".", 1)
                path = f"{stem}.{m}.{ext}"
            else:
                path = f"{dump_base}.{m}.yaml"
            save_dataclass_yaml(cfg, path)
            logging.info(f"Saved {m} config to {path}")
        did_io = True

    return did_io


def main(argv: Optional[List[str]] = None) -> int:
    """PG-SUI CLI main entry point.

    The CLI supports running multiple imputation models on a single input file, with configuration handled via presets, YAML files, and CLI flags.

    Args:
        argv (Optional[List[str]]): List of CLI args (default: sys.argv[1:]).

    Returns:
        int: Exit code (0=success, 2=argparse error, 1=other error).
    """
    parser = argparse.ArgumentParser(
        prog="pg-sui",
        description="Run PG-SUI imputation models on an input file. Handle configuration via presets, YAML, and CLI flags. The default is to run all models. The input file can be in VCF, PHYLIP, or GENEPOP format. Outputs include imputed genotype files and performance summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s [options]",
    )

    # ----------------------------- Required I/O ----------------------------- #
    parser.add_argument(
        "--input",
        default=argparse.SUPPRESS,
        help="Path to input file (VCF/PHYLIP/GENEPOP). VCF file can be bgzipped or uncompressed.",
    )
    parser.add_argument(
        "--format",
        choices=(
            "infer",
            "vcf",
            "vcf.gz",
            "phy",
            "phylip",
            "str",
            "structure",
            "genepop",
            "gen",
        ),
        default=argparse.SUPPRESS,
        help="Input format. If 'infer', deduced from file extension. The default is 'infer'. Supported formats: VCF ('.vcf', '.vcf.gz'), PHYLIP ('.phy', '.phylip'), GENEPOP ('.genepop', '.gen').",
    )
    # Back-compat: --vcf retained; if both provided, --input wins.
    parser.add_argument(
        "--vcf",
        default=argparse.SUPPRESS,
        help="Path to input VCF file. Can be bgzipped or uncompressed. (Deprecated; use --input instead.)",
    )
    parser.add_argument(
        "--popmap",
        default=argparse.SUPPRESS,
        help="Path to population map file. This is a two-column tab-delimited file with sample IDs and population IDs. If not provided, no population info is used.",
    )
    parser.add_argument(
        "--treefile",
        default=argparse.SUPPRESS,
        help="Path to phylogenetic tree file. Can be in Newick (recommended) or Nexus format. Used with --qmatrix and --siterates.",
    )
    parser.add_argument(
        "--qmatrix",
        default=argparse.SUPPRESS,
        help="Path to IQ-TREE output file (has .iqtree extension) that contains Rate Matrix Q. Used with --treefile and --siterates.",
    )
    parser.add_argument(
        "--siterates",
        default=argparse.SUPPRESS,
        help="Path to SNP site rates file (has .rate extension and can be produced with IQ-TREE). Used with --treefile and --qmatrix.",
    )
    parser.add_argument(
        "--prefix",
        default=argparse.SUPPRESS,
        help="Output file prefix. If not provided, defaults to the input file stem.",
    )

    # ---------------------- Generic Config Inputs -------------------------- #
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="YAML config for config-driven models (Autoencoder, VAE). Overrides preset and defaults.",
    )
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "thorough"),
        default=argparse.SUPPRESS,  # <-- no default; optional
        help="If provided, initialize config(s) from this preset; otherwise start from dataclass defaults.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=argparse.SUPPRESS,
        help="Dot-key overrides, e.g. --set model.latent_dim=4 --set train.epochs=100. Applies to all models.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective config(s) and exit.",
    )
    parser.add_argument(
        "--dump-config",
        default=argparse.SUPPRESS,
        help="Write effective config(s) YAML to this path (multi-model gets suffixed).",
    )

    # ------------------------------ Toggles -------------------------------- #
    parser.add_argument(
        "--tune",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable hyperparameter tuning (if supported by model). Uses Optuna to optimize hyperparameters.",
    )
    parser.add_argument(
        "--tune-n-trials",
        type=int,
        default=argparse.SUPPRESS,
        help="Optuna trials when --tune is set.",
    )
    parser.add_argument(
        "--tune-metrics",
        nargs="+",
        type=str,
        default=argparse.SUPPRESS,
        help="Metric(s) to optimize during hyperparameter tuning. Applies to all models that support tuning. Choices: accuracy, f1, precision, recall, pr_macro, roc_auc, average_precision, mcc, jaccard. You can supply multiple metrics for multi-objective Optuna tuning by separating with spaces (Default: f1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Batch size for NN-based models.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=argparse.SUPPRESS,
        help="Compute device for NN-based models.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=argparse.SUPPRESS,
        help="Parallel workers for various steps.",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf", "svg", "jpg", "jpeg"),
        default=argparse.SUPPRESS,
        help="Figure format for model plots. Choices: png, pdf, svg, jpg, jpeg.",
    )
    parser.add_argument(
        "--disable-plotting",
        action="store_true",
        default=False,
        help="Disable plotting for all models. Overrides any config settings enabling plotting.",
    )

    parser.add_argument(
        "--load-best-params",
        action="store_true",
        default=False,
        help=(
            "Load best hyperparameters from a previous run's best_parameters.json (or tuning best_tuned_parameters.json) for each selected model and apply them to the model configs. This forces tuning OFF."
        ),
    )
    parser.add_argument(
        "--best-params-prefix",
        default=argparse.SUPPRESS,
        help=(
            "Prefix of the PREVIOUS run to load best parameters from. If omitted, uses the current --prefix (or input stem)."
        ),
    )

    # ------------------------- Simulation Controls ------------------------ #
    parser.add_argument(
        "--sim-strategy",
        choices=SIM_STRATEGY_CHOICES,
        default=argparse.SUPPRESS,
        help="Override the missing-data simulation strategy for all config-driven models.",
    )
    parser.add_argument(
        "--sim-prop",
        type=float,
        default=argparse.SUPPRESS,
        help="Override the proportion of observed entries to mask during simulation (0-1).",
    )

    # --------------------------- Seed & logging ---------------------------- #
    parser.add_argument(
        "--seed",
        default=argparse.SUPPRESS,
        help="Random seed: 'random', 'deterministic', or an integer. Default is 'random'.",
    )

    # ----------------------------- Logging --------------------------------- #
    parser.add_argument("--verbose", action="store_true", help="Info-level logging.")
    parser.add_argument("--debug", action="store_true", help="Debug-level logging.")
    parser.add_argument(
        "--log-file", default=argparse.SUPPRESS, help="Also write logs to a file."
    )

    # ---------------------------- Data filtering --------------------------- #
    parser.add_argument(
        "--include-pops",
        nargs="+",
        default=argparse.SUPPRESS,
        help="Optional list of population IDs to include.",
    )
    parser.add_argument(
        "--force-popmap",
        action="store_true",
        default=False,
        help="Force use of provided popmap even if samples don't match exactly. This will drop samples not in the popmap and vice versa.",
    )

    # -------------------------- STRUCTURE options ------------------------- #
    parser.add_argument(
        "--structure-has-popids",
        action="store_true",
        default=False,
        help="STRUCTURE only: second column contains population IDs.",
    )
    parser.add_argument(
        "--structure-has-marker-names",
        action="store_true",
        default=False,
        help="STRUCTURE only: first row contains marker names.",
    )
    parser.add_argument(
        "--structure-allele-start-col",
        type=int,
        default=argparse.SUPPRESS,
        help="STRUCTURE only: zero-based column index where alleles begin.",
    )
    parser.add_argument(
        "--structure-allele-encoding",
        type=_parse_allele_encoding,
        default=argparse.SUPPRESS,
        help="STRUCTURE only: allele encoding mapping as JSON or Python dict.",
    )

    # ---------------------------- Model selection -------------------------- #
    parser.add_argument(
        "--models",
        nargs="+",
        default=argparse.SUPPRESS,
        help=(
            "Which models to run. Specify each model separated by a space. Choices: ImputeUBP ImputeNLPCA ImputeVAE ImputeAutoencoder ImputeMostFrequent ImputeRefAllele (Default is all models)."
        ),
    )

    # -------------------------- MultiQC integration ------------------------ #
    parser.add_argument(
        "--disable-multiqc",
        action="store_true",
        default=False,
        help=(
            "Disable MultiQC report generation after imputation. By default, a MultiQC report is generated unless this flag is set."
        ),
    )
    parser.add_argument(
        "--multiqc-title",
        default=argparse.SUPPRESS,
        help="Optional title for the MultiQC report (default: 'PG-SUI MultiQC Report - <prefix>'). ",
    )
    parser.add_argument(
        "--multiqc-output-dir",
        default=argparse.SUPPRESS,
        help="Optional output directory for the MultiQC report (default: '<prefix>_output/multiqc'). This directory will be created if it does not exist.",
    )
    parser.add_argument(
        "--multiqc-overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing MultiQC report if present. If not set and a report exists, an integer suffix will be added to avoid overwriting. NOTE: if running multiple times with this flag, it may append multiple suffixes to avoid overwriting previous reports.",
    )

    # ------------------------------ Safety/UX ------------------------------ #
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and load data, but skip model training. Useful for testing I/O and configs.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print PG-SUI version and exit."
    )

    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        _print_version()
        return 0

    # Logging (verbose default is False unless passed)
    _configure_logging(
        verbose=getattr(args, "verbose", False),
        log_file=getattr(args, "log_file", None),
    )

    logging.info("Starting PG-SUI imputation...")
    _print_version()

    # Models selection (default to all if not explicitly provided)
    try:
        selected_models = _parse_models(getattr(args, "models", ()))
    except argparse.ArgumentTypeError as e:
        logging.error(str(e))
        parser.error(str(e))
        return 2

    # Input resolution
    input_path = getattr(args, "input", None)
    if input_path is None and hasattr(args, "vcf"):
        input_path = args.vcf
        if not hasattr(args, "format"):
            setattr(args, "format", "vcf")

    if input_path is None:
        logging.error("You must provide --input (or legacy --vcf).")
        parser.error("You must provide --input (or legacy --vcf).")
        return 2

    fmt = getattr(args, "format", "infer")

    if fmt == "infer":
        if input_path.endswith((".vcf", ".vcf.gz")):
            fmt_final = "vcf"
        elif input_path.endswith((".phy", ".phylip")):
            fmt_final = "phylip"
        elif input_path.endswith((".genepop", ".gen")):
            fmt_final = "genepop"
        elif input_path.endswith((".str", ".stru", ".structure")):
            fmt_final = "structure"
        else:
            logging.error(
                "Could not infer input format from file extension. Please provide --format."
            )
            parser.error(
                "Could not infer input format from file extension. Please provide --format."
            )
            return 2
    else:
        fmt_final = cast(
            Literal[
                "vcf",
                "vcf.gz",
                "phy",
                "phylip",
                "genepop",
                "gen",
                "structure",
                "str",
            ],
            fmt,
        )

    fmt_final = _normalize_input_format(fmt_final)

    popmap_path = getattr(args, "popmap", None)
    include_pops = getattr(args, "include_pops", None)
    force_popmap = bool(getattr(args, "force_popmap", False))
    structure_has_popids = bool(getattr(args, "structure_has_popids", False))
    structure_has_marker_names = bool(
        getattr(args, "structure_has_marker_names", False)
    )
    structure_allele_start_col = getattr(args, "structure_allele_start_col", None)
    structure_allele_encoding = getattr(args, "structure_allele_encoding", None)

    # Canonical prefix for this run (used for outputs and MultiQC)
    prefix: str = getattr(args, "prefix", str(Path(input_path).stem))
    # Ensure downstream config building sees the resolved prefix even if
    # --prefix was not provided.
    setattr(args, "prefix", prefix)

    treefile, qmatrix, siterates = _resolve_tree_paths(args)
    setattr(args, "treefile", treefile)
    setattr(args, "qmatrix", qmatrix)
    setattr(args, "siterates", siterates)

    if any(x is not None for x in (treefile, qmatrix, siterates)):
        if not all(x is not None for x in (treefile, qmatrix, siterates)):
            logging.error(
                "--treefile, --qmatrix, and --siterates must all be provided together or they should all be omitted."
            )
            parser.error(
                "--treefile, --qmatrix, and --siterates must all be provided together or they should all be omitted."
            )
            return 2

    # Load genotype data
    gd, tp = build_genotype_data(
        input_path=input_path,
        fmt=cast(
            Literal[
                "vcf",
                "vcf.gz",
                "phy",
                "phylip",
                "genepop",
                "gen",
                "structure",
                "str",
            ],
            fmt_final,
        ),
        popmap_path=popmap_path,
        treefile=treefile,
        qmatrix=qmatrix,
        siterates=siterates,
        force_popmap=force_popmap,
        include_pops=include_pops,
        debug=getattr(args, "debug", False),
        plot_format=getattr(args, "plot_format", "pdf"),
        structure_has_popids=structure_has_popids,
        structure_has_marker_names=structure_has_marker_names,
        structure_allele_start_col=structure_allele_start_col,
        structure_allele_encoding=structure_allele_encoding,
    )

    if getattr(args, "dry_run", False):
        logging.info("Dry run complete. Exiting without training models.")
        return 0

    # ---------------- Build config(s) per selected model ------------------- #
    cfgs_by_model: Dict[str, Any] = {
        m: _build_effective_config_for_model(m, args) for m in selected_models
    }

    needs_tree = any(
        _config_needs_tree(cfg) for cfg in cfgs_by_model.values() if cfg is not None
    )
    if needs_tree and not all(x is not None for x in (treefile, qmatrix, siterates)):
        logging.error(
            "Nonrandom simulated missingness requires --treefile, --qmatrix, and --siterates."
        )
        parser.error(
            "Nonrandom simulated missingness requires --treefile, --qmatrix, and --siterates."
        )
        return 2
    if needs_tree and tp is None:
        logging.error(
            "Tree parser was not initialized for nonrandom simulation. "
            "Please verify --treefile, --qmatrix, and --siterates."
        )
        parser.error(
            "Tree parser was not initialized for nonrandom simulation. "
            "Please verify --treefile, --qmatrix, and --siterates."
        )
        return 2

    # Maybe print/dump configs and exit
    if _maybe_print_or_dump_configs(cfgs_by_model, args):
        return 0

    # ------------------------- Model Builders ------------------------------ #
    def build_impute_vae():
        cfg = cfgs_by_model.get("ImputeVAE")
        if cfg is None:
            cfg = (
                VAEConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else VAEConfig()
            )
        return ImputeVAE(
            genotype_data=gd,
            tree_parser=tp,
            config=cfg,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    def build_impute_autoencoder():
        cfg = cfgs_by_model.get("ImputeAutoencoder")
        if cfg is None:
            cfg = (
                AutoencoderConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else AutoencoderConfig()
            )
        return ImputeAutoencoder(
            genotype_data=gd,
            tree_parser=tp,
            config=cfg,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    def build_impute_ubp():
        cfg = cfgs_by_model.get("ImputeUBP")
        if cfg is None:
            cfg = (
                UBPConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else UBPConfig()
            )
        return ImputeUBP(
            genotype_data=gd,
            tree_parser=tp,
            config=cfg,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    def build_impute_nlpca():
        cfg = cfgs_by_model.get("ImputeNLPCA")
        if cfg is None:
            cfg = (
                NLPCAConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else NLPCAConfig()
            )
        return ImputeNLPCA(
            genotype_data=gd,
            tree_parser=tp,
            config=cfg,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    def build_impute_mostfreq():
        cfg = cfgs_by_model.get("ImputeMostFrequent")
        if cfg is None:
            cfg = (
                MostFrequentConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else MostFrequentConfig()
            )
        return ImputeMostFrequent(
            gd,
            tree_parser=tp,
            config=cfg,
            simulate_missing=True,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    def build_impute_refallele():
        cfg = cfgs_by_model.get("ImputeRefAllele")
        if cfg is None:
            cfg = (
                RefAlleleConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else RefAlleleConfig()
            )
        return ImputeRefAllele(
            gd,
            tree_parser=tp,
            config=cfg,
            simulate_missing=True,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    model_builders = {
        "ImputeUBP": build_impute_ubp,
        "ImputeNLPCA": build_impute_nlpca,
        "ImputeVAE": build_impute_vae,
        "ImputeAutoencoder": build_impute_autoencoder,
        "ImputeMostFrequent": build_impute_mostfreq,
        "ImputeRefAllele": build_impute_refallele,
    }

    logging.info(f"Selected models: {', '.join(selected_models)}")
    for name in selected_models:
        logging.info("")
        logging.info("=" * 60)
        logging.info("")
        logging.info(f"Processing model: {name} ...")
        X_imputed = run_model_safely(name, model_builders[name], warn_only=False)
        gd_imp = gd.copy()
        gd_imp.snp_data = X_imputed

        if name in {"ImputeUBP", "ImputeNLPCA", "ImputeVAE", "ImputeAutoencoder"}:
            family = "Unsupervised"
        elif name in {"ImputeMostFrequent", "ImputeRefAllele"}:
            family = "Deterministic"
        elif name in {"ImputeHistGradientBoosting", "ImputeRandomForest"}:
            family = "Supervised"
        else:
            logging.error(f"Unknown model family for {name}")
            raise ValueError(f"Unknown model family for {name}")

        pth = Path(f"{prefix}_output/{family}/imputed/{name}")
        pth.mkdir(parents=True, exist_ok=True)

        logging.info(f"Writing imputed VCF for {name} to {pth} ...")

        if fmt_final == "vcf":
            gd_imp.write_vcf(pth / f"{name.lower()}_imputed.vcf.gz")
        elif fmt_final == "phylip":
            gd_imp.write_phylip(pth / f"{name.lower()}_imputed.phy")
        elif fmt_final == "genepop":
            gd_imp.write_genepop(pth / f"{name.lower()}_imputed.gen")
        else:
            logging.warning(
                f"Output format {fmt_final} not supported for imputed data export."
            )

        logging.info("")
        logging.info(f"Successfully finished imputation for model: {name}!")
        logging.info("")
        logging.info("=" * 60)

    logging.info(f"All requested models processed for input: {input_path}")

    disable_mqc = bool(getattr(args, "disable_multiqc", False))

    if disable_mqc:
        logging.info("MultiQC report generation disabled via --disable-multiqc.")
        logging.info("PG-SUI imputation run complete!")
        return 0

    # -------------------------- MultiQC builder ---------------------------- #

    mqc_output_dir = getattr(args, "multiqc_output_dir", f"{prefix}_output/multiqc")
    mqc_title = getattr(args, "multiqc_title", f"PG-SUI MultiQC Report - {prefix}")
    overwrite = bool(getattr(args, "multiqc_overwrite", False))

    logging.info(
        f"Building MultiQC report in '{mqc_output_dir}' (title={mqc_title}, overwrite={overwrite})..."
    )

    try:
        SNPioMultiQC.build(
            prefix=prefix,
            output_dir=mqc_output_dir,
            title=mqc_title,
            overwrite=overwrite,
        )
        logging.info("MultiQC report successfully built.")
    except Exception as exc2:
        logging.error(f"Failed to build MultiQC report: {exc2}", exc_info=True)

    logging.info("PG-SUI imputation run complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
