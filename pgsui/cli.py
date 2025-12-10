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
python cli.py --vcf data.vcf.gz --popmap pops.popmap --prefix run1
python cli.py --vcf data.vcf.gz --popmap pops.popmap --prefix tuned --tune
python cli.py --vcf data.vcf.gz --popmap pops.popmap --prefix demo \
    --models ImputeUBP ImputeVAE ImputeMostFrequent --seed deterministic --verbose
python cli.py --vcf data.vcf.gz --popmap pops.popmap --prefix subset \
    --include-pops EA GU TT ON --device cpu
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import logging
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

from snpio import GenePopReader, PhylipReader, SNPioMultiQC, TreeParser, VCFReader

from pgsui import (
    AutoencoderConfig,
    ImputeAutoencoder,
    ImputeMostFrequent,
    ImputeNLPCA,
    ImputeRefAllele,
    ImputeUBP,
    ImputeVAE,
    MostFrequentConfig,
    NLPCAConfig,
    RefAlleleConfig,
    UBPConfig,
    VAEConfig,
)
from pgsui.data_processing.config import (
    apply_dot_overrides,
    dataclass_to_yaml,
    load_yaml_to_dataclass,
    save_dataclass_yaml,
)

# Canonical model order used everywhere (default and subset ordering)
MODEL_ORDER: Tuple[str, ...] = (
    "ImputeUBP",
    "ImputeVAE",
    "ImputeAutoencoder",
    "ImputeNLPCA",
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


def _args_to_cli_overrides(args: argparse.Namespace) -> dict:
    """Convert explicitly provided CLI flags into config dot-overrides."""
    overrides: dict = {}

    # IO / top-level controls
    if hasattr(args, "prefix") and args.prefix is not None:
        overrides["io.prefix"] = args.prefix
    else:
        # Note: we don't know input_path here; prefix default is handled later.
        # This fallback is preserved to avoid changing semantics.
        if hasattr(args, "vcf"):
            overrides["io.prefix"] = str(Path(args.vcf).stem)

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

    # Simulation overrides (shared across config-driven models)
    if hasattr(args, "sim_strategy"):
        overrides["sim.sim_strategy"] = args.sim_strategy
    if hasattr(args, "sim_prop"):
        overrides["sim.sim_prop"] = float(args.sim_prop)
    if hasattr(args, "simulate_missing"):
        overrides["sim.simulate_missing"] = bool(args.simulate_missing)

    # Tuning
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
    fmt: Literal["vcf", "vcf.gz", "phy", "phylip", "genepop", "gen"],
    popmap_path: str | None,
    treefile: str | None,
    qmatrix: str | None,
    siterates: str | None,
    force_popmap: bool,
    verbose: bool,
    include_pops: List[str] | None,
    plot_format: Literal["pdf", "png", "jpg", "jpeg"],
):
    """Load genotype data from heterogeneous inputs."""
    logging.info(f"Loading {fmt.upper()} and popmap data...")

    kwargs = {
        "filename": input_path,
        "popmapfile": popmap_path,
        "force_popmap": force_popmap,
        "verbose": verbose,
        "include_pops": include_pops if include_pops else None,
        "prefix": f"snpio_{Path(input_path).stem}",
        "plot_format": plot_format,
    }

    if fmt == "vcf":
        gd = VCFReader(**kwargs)
    elif fmt == "phylip":
        gd = PhylipReader(**kwargs)
    elif fmt == "genepop":
        gd = GenePopReader(**kwargs)
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

    # 3) Explicit CLI flags overlay YAML.
    cli_overrides = _args_to_cli_overrides(args)
    if cli_overrides:
        cfg = apply_dot_overrides(cfg, cli_overrides)

    # 4) --set has highest precedence.
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
    parser = argparse.ArgumentParser(
        prog="pg-sui",
        description="Run PG-SUI imputation models on an input file. Handle configuration via presets, YAML, and CLI flags. The default is to run all models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="%(prog)s [options]",
    )

    # ----------------------------- Required I/O ----------------------------- #
    parser.add_argument(
        "--input",
        default=argparse.SUPPRESS,
        help="Path to input file (VCF/PHYLIP/STRUCTURE/GENEPOP).",
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
        help="Input format. If 'infer', deduced from file extension. The default is 'infer'.",
    )
    # Back-compat: --vcf retained; if both provided, --input wins.
    parser.add_argument(
        "--vcf",
        default=argparse.SUPPRESS,
        help="Path to input VCF file. Can be bgzipped or uncompressed.",
    )
    parser.add_argument(
        "--popmap",
        default=argparse.SUPPRESS,
        help="Path to population map file. This is a two-column tab-delimited file with sample IDs and population IDs.",
    )
    parser.add_argument(
        "--treefile",
        default=argparse.SUPPRESS,
        help="Path to phylogenetic tree file. Can be in Newick (recommended) or Nexus format.",
    )
    parser.add_argument(
        "--qmatrix",
        default=argparse.SUPPRESS,
        help="Path to IQ-TREE output file (has .iqtree extension) that contains Rate Matrix Q. Used with --treefile and --siterates.",
    )
    parser.add_argument(
        "--siterates",
        default=argparse.SUPPRESS,
        help="Path to SNP site rates file (has .rate extension). Used with --treefile and --qmatrix.",
    )
    parser.add_argument(
        "--prefix",
        default=argparse.SUPPRESS,
        help="Output file prefix.",
    )

    # ---------------------- Generic Config Inputs -------------------------- #
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="YAML config for config-driven models (NLPCA/UBP/Autoencoder/VAE).",
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
        help="Dot-key overrides, e.g. --set model.latent_dim=4",
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
        help="Enable hyperparameter tuning (if supported).",
    )
    parser.add_argument(
        "--tune-n-trials",
        type=int,
        default=argparse.SUPPRESS,
        help="Optuna trials when --tune is set.",
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
        help="Figure format for model plots.",
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
    parser.add_argument(
        "--simulate-missing",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable missing-data simulation regardless of preset/config (when provided).",
    )

    # --------------------------- Seed & logging ---------------------------- #
    parser.add_argument(
        "--seed",
        default=argparse.SUPPRESS,
        help="Random seed: 'random', 'deterministic', or an integer.",
    )
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
        help="Require popmap (error if absent).",
    )

    # ---------------------------- Model selection -------------------------- #
    parser.add_argument(
        "--models",
        nargs="+",
        default=argparse.SUPPRESS,
        help=(
            "Which models to run. Choices: ImputeUBP ImputeVAE ImputeAutoencoder ImputeNLPCA ImputeMostFrequent ImputeRefAllele. Default is all."
        ),
    )

    # -------------------------- MultiQC integration ------------------------ #
    parser.add_argument(
        "--multiqc",
        action="store_true",
        help=(
            "Build a MultiQC HTML report at the end of the run, combining SNPio and PG-SUI plots (requires SNPio's MultiQC module)."
        ),
    )
    parser.add_argument(
        "--multiqc-title",
        default=argparse.SUPPRESS,
        help="Optional title for the MultiQC report (default: 'PG-SUI MultiQC Report - <prefix>').",
    )
    parser.add_argument(
        "--multiqc-output-dir",
        default=argparse.SUPPRESS,
        help="Optional output directory for the MultiQC report (default: '<prefix>_output/multiqc').",
    )
    parser.add_argument(
        "--multiqc-overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing MultiQC report if present.",
    )

    # ------------------------------ Safety/UX ------------------------------ #
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and load data, but skip model training.",
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
        parser.error(str(e))
        return 2

    # Input resolution
    input_path = getattr(args, "input", None)
    if input_path is None and hasattr(args, "vcf"):
        input_path = args.vcf
        if not hasattr(args, "format"):
            setattr(args, "format", "vcf")

    if input_path is None:
        parser.error("You must provide --input (or legacy --vcf).")
        return 2

    fmt: Literal["infer", "vcf", "vcf.gz", "phy", "phylip", "genepop", "gen"] = getattr(
        args, "format", "infer"
    )

    if fmt == "infer":
        if input_path.endswith((".vcf", ".vcf.gz")):
            fmt_final = "vcf"
        elif input_path.endswith((".phy", ".phylip")):
            fmt_final = "phylip"
        elif input_path.endswith((".genepop", ".gen")):
            fmt_final = "genepop"
        else:
            parser.error(
                "Could not infer input format from file extension. Please provide --format."
            )
            return 2
    else:
        fmt_final = fmt

    popmap_path = getattr(args, "popmap", None)
    include_pops = getattr(args, "include_pops", None)
    verbose_flag = getattr(args, "verbose", False)
    force_popmap = bool(getattr(args, "force_popmap", False))

    # Canonical prefix for this run (used for outputs and MultiQC)
    prefix: str = getattr(args, "prefix", str(Path(input_path).stem))

    treefile = getattr(args, "treefile", None)
    qmatrix = getattr(args, "qmatrix", None)
    siterates = getattr(args, "siterates", None)

    if any(x is not None for x in (treefile, qmatrix, siterates)):
        if not all(x is not None for x in (treefile, qmatrix, siterates)):
            parser.error(
                "--treefile, --qmatrix, and --siterates must all be provided together or they should all be omitted."
            )
            return 2

    # Load genotype data
    gd, tp = build_genotype_data(
        input_path=input_path,
        fmt=fmt_final,
        popmap_path=popmap_path,
        treefile=treefile,
        qmatrix=qmatrix,
        siterates=siterates,
        force_popmap=force_popmap,
        verbose=verbose_flag,
        include_pops=include_pops,
        plot_format=getattr(args, "plot_format", "pdf"),
    )

    if getattr(args, "dry_run", False):
        logging.info("Dry run complete. Exiting without training models.")
        return 0

    # ---------------- Build config(s) per selected model ------------------- #
    cfgs_by_model: Dict[str, Any] = {
        m: _build_effective_config_for_model(m, args) for m in selected_models
    }

    # Maybe print/dump configs and exit
    if _maybe_print_or_dump_configs(cfgs_by_model, args):
        return 0

    # ------------------------- Model Builders ------------------------------ #
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
            simulate_missing=cfg.sim.simulate_missing,
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
            simulate_missing=cfg.sim.simulate_missing,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

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
            simulate_missing=cfg.sim.simulate_missing,
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
            simulate_missing=cfg.sim.simulate_missing,
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
            simulate_missing=cfg.sim.simulate_missing,
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
            simulate_missing=cfg.sim.simulate_missing,
            sim_strategy=cfg.sim.sim_strategy,
            sim_prop=cfg.sim.sim_prop,
            sim_kwargs=cfg.sim.sim_kwargs,
        )

    model_builders = {
        "ImputeUBP": build_impute_ubp,
        "ImputeVAE": build_impute_vae,
        "ImputeAutoencoder": build_impute_autoencoder,
        "ImputeNLPCA": build_impute_nlpca,
        "ImputeMostFrequent": build_impute_mostfreq,
        "ImputeRefAllele": build_impute_refallele,
    }

    logging.info(f"Selected models: {', '.join(selected_models)}")
    for name in selected_models:
        X_imputed = run_model_safely(name, model_builders[name], warn_only=False)
        gd_imp = gd.copy()
        gd_imp.snp_data = X_imputed

        if name in {"ImputeUBP", "ImputeVAE", "ImputeAutoencoder", "ImputeNLPCA"}:
            family = "Unsupervised"
        elif name in {"ImputeMostFrequent", "ImputeRefAllele"}:
            family = "Deterministic"
        elif name in {"ImputeHistGradientBoosting", "ImputeRandomForest"}:
            family = "Supervised"
        else:
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

    logging.info("All requested models processed.")

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
    except Exception as exc2:  # pragma: no cover
        logging.error(f"Failed to build MultiQC report: {exc2}", exc_info=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
