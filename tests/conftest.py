from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _configure_matplotlib_cache(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Ensure Matplotlib uses a writable cache during tests."""
    cache_dir = tmp_path_factory.mktemp("mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


@pytest.fixture(scope="session")
def example_vcf_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "pgsui"
        / "example_data"
        / "vcf_files"
        / "phylogen_subset14K.vcf.gz"
    )


@pytest.fixture(scope="session")
def example_popmap_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "pgsui"
        / "example_data"
        / "popmaps"
        / "phylogen_nomx.popmap"
    )


@pytest.fixture(scope="session")
def example_genotype_data(
    tmp_path_factory: pytest.TempPathFactory,
    example_vcf_path: Path,
    example_popmap_path: Path,
):
    """Load the bundled example dataset via SNPio's VCFReader."""
    snpio = pytest.importorskip("snpio")
    from snpio import VCFReader

    workdir = tmp_path_factory.mktemp("pgsui_runs")
    reader = VCFReader(
        filename=str(example_vcf_path),
        popmapfile=str(example_popmap_path),
        prefix=str(workdir / "example"),
        force_popmap=True,
        plot_format="pdf",
    )
    return reader
