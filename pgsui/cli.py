#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PG-SUI Imputation CLI

Argument-precedence model:
    code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set k=v

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
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from snpio import VCFReader

from pgsui import (
    ImputeAutoencoder,
    ImputeMostFrequent,
    ImputeNLPCA,
    ImputeRefAllele,
    ImputeUBP,
    ImputeVAE,
)
from pgsui.data_processing.config import (
    apply_dot_overrides,
    dataclass_to_yaml,
    load_yaml_to_dataclass,
    save_dataclass_yaml,
)
from pgsui.data_processing.containers import (
    AutoencoderConfig,
    MostFrequentConfig,
    NLPCAConfig,
    RefAlleleConfig,
    UBPConfig,
    VAEConfig,
)


# ----------------------------- CLI Utilities ----------------------------- #
def _configure_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    """Configure root logger.

    Args:
        verbose: If True, DEBUG; else INFO.
        log_file: Optional file to tee logs to.
    """
    level = logging.DEBUG if verbose else logging.INFO
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
    """Validate and canonicalize model names."""
    valid = {
        "ImputeUBP",
        "ImputeVAE",
        "ImputeAutoencoder",
        "ImputeNLPCA",
        "ImputeMostFrequent",
        "ImputeRefAllele",
    }
    selected = tuple(models) if models else tuple(valid)
    unknown = [m for m in selected if m not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown model(s): {unknown}. Valid options: {sorted(valid)}"
        )
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
    if hasattr(args, "verbose"):
        overrides["io.verbose"] = bool(args.verbose)
    if hasattr(args, "n_jobs"):
        overrides["io.n_jobs"] = int(args.n_jobs)
    if hasattr(args, "seed"):
        overrides["io.seed"] = _parse_seed(args.seed)

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

    # Tuning
    if hasattr(args, "tune"):
        overrides["tune.enabled"] = bool(args.tune)
    if hasattr(args, "tune_n_trials"):
        overrides["tune.n_trials"] = int(args.tune_n_trials)

    return overrides


# ------------------------------ Core Runner ------------------------------ #
def build_genotype_data(
    vcf_path: str,
    popmap_path: str | None,
    force_popmap: bool,
    verbose: bool,
    include_pops: List[str] | None,
) -> VCFReader:
    """Load genotype data from VCF/popmap."""
    logging.info("Loading VCF and popmap data...")
    gd = VCFReader(
        filename=vcf_path,
        popmapfile=popmap_path,
        force_popmap=force_popmap,
        verbose=verbose,
        include_pops=include_pops if include_pops else None,
        prefix=f"snpio_{Path(vcf_path).stem}",
    )
    logging.info("Loaded genotype data.")
    return gd


def run_model_safely(model_name: str, builder, *, warn_only: bool = True) -> None:
    """Run model builder + fit/transform with error isolation."""
    logging.info("▶ Running %s ...", model_name)
    try:
        model = builder()
        model.fit()
        X_imputed = model.transform()
        logging.info("✓ %s completed.", model_name)
        return X_imputed
    except Exception as e:
        if warn_only:
            logging.warning("⚠ %s failed: %s", model_name, e, exc_info=True)
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
        logging.info("Initialized %s from '%s' preset.", model_name, preset_name)
    else:
        logging.info("Initialized %s from dataclass defaults (no preset).", model_name)

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
            "Loaded YAML config for %s from %s (ignored 'preset' in YAML if present).",
            model_name,
            yaml_path,
        )

    # 3) Explicit CLI flags overlay YAML (boss's boss).
    cli_overrides = _args_to_cli_overrides(args)
    if cli_overrides:
        cfg = apply_dot_overrides(cfg, cli_overrides)

    # 4) --set has highest precedence (supreme boss).
    user_overrides = _parse_overrides(getattr(args, "set", []))
    if user_overrides:
        cfg = apply_dot_overrides(cfg, user_overrides)

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
            logging.info("Saved %s config to %s", m, path)
        did_io = True

    return did_io


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pgsui-cli",
        description="Run PG-SUI imputation models on a VCF with minimal fuss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----------------------------- Required I/O ----------------------------- #
    parser.add_argument("--vcf", required=True, help="Path to input VCF(.gz) file.")
    parser.add_argument(
        "--popmap", default=argparse.SUPPRESS, help="Path to population map file."
    )
    parser.add_argument(
        "--prefix",
        default=argparse.SUPPRESS,
        help="Run/output prefix; overrides config if provided.",
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
        choices=("png", "pdf", "svg"),
        default=argparse.SUPPRESS,
        help="Figure format for model plots.",
    )

    # --------------------------- Seed & logging ---------------------------- #
    parser.add_argument(
        "--seed",
        default=argparse.SUPPRESS,
        help="Random seed: 'random', 'deterministic', or an integer.",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
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
            "Which models to run. Choices: "
            "ImputeUBP ImputeVAE ImputeAutoencoder ImputeNLPCA "
            "ImputeMostFrequent ImputeRefAllele. Default is all."
        ),
    )

    # ------------------------------ Safety/UX ------------------------------ #
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and load data, but skip model training.",
    )

    args = parser.parse_args(argv)

    # Logging (verbose default is False unless passed)
    _configure_logging(
        verbose=getattr(args, "verbose", False),
        log_file=getattr(args, "log_file", None),
    )

    # Models selection (default to all if not explicitly provided)
    try:
        selected_models = _parse_models(getattr(args, "models", ()))
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
        return 2

    # Data I/O flags
    vcf_path = args.vcf
    popmap_path = getattr(args, "popmap", None)
    include_pops = getattr(args, "include_pops", None)
    verbose_flag = getattr(args, "verbose", False)
    force_popmap = bool(getattr(args, "force_popmap", False))

    # Load genotype data
    gd = build_genotype_data(
        vcf_path=vcf_path,
        popmap_path=popmap_path,
        force_popmap=force_popmap,
        verbose=verbose_flag,
        include_pops=include_pops,
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
        return ImputeUBP(genotype_data=gd, config=cfg)

    def build_impute_nlpca():
        cfg = cfgs_by_model.get("ImputeNLPCA")
        if cfg is None:
            cfg = (
                NLPCAConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else NLPCAConfig()
            )
        return ImputeNLPCA(genotype_data=gd, config=cfg)

    def build_impute_vae():
        cfg = cfgs_by_model.get("ImputeVAE")
        if cfg is None:
            cfg = (
                VAEConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else VAEConfig()
            )
        return ImputeVAE(genotype_data=gd, config=cfg)

    def build_impute_autoencoder():
        cfg = cfgs_by_model.get("ImputeAutoencoder")
        if cfg is None:
            cfg = (
                AutoencoderConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else AutoencoderConfig()
            )
        return ImputeAutoencoder(genotype_data=gd, config=cfg)

    def build_impute_mostfreq():
        cfg = cfgs_by_model.get("ImputeMostFrequent")
        if cfg is None:
            cfg = (
                MostFrequentConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else MostFrequentConfig()
            )
        return ImputeMostFrequent(gd, config=cfg)

    def build_impute_refallele():
        cfg = cfgs_by_model.get("ImputeRefAllele")
        if cfg is None:
            cfg = (
                RefAlleleConfig.from_preset(args.preset)
                if hasattr(args, "preset")
                else RefAlleleConfig()
            )
        return ImputeRefAllele(gd, config=cfg)

    model_builders = {
        "ImputeUBP": build_impute_ubp,
        "ImputeVAE": build_impute_vae,
        "ImputeAutoencoder": build_impute_autoencoder,
        "ImputeNLPCA": build_impute_nlpca,
        "ImputeMostFrequent": build_impute_mostfreq,
        "ImputeRefAllele": build_impute_refallele,
    }

    logging.info("Selected models: %s", ", ".join(selected_models))
    for name in selected_models:
        X_imputed = run_model_safely(name, model_builders[name], warn_only=True)
        gd_imp = gd.copy()
        gd_imp.snp_data = X_imputed

        pth = Path(f"psgui_output/{Path(vcf_path).stem}_output/imputed")
        pth.mkdir(parents=True, exist_ok=True)
        logging.info("Writing imputed VCF for %s to %s ...", name, pth)
        gd_imp.write_vcf(pth / f"{name.lower()}.vcf.gz")

    logging.info("All requested models processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
