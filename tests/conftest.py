from __future__ import annotations

from curses.ascii import alt
import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Compatibility shims for older snpio versions (missing SNPioMultiQC)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - defensive guard
    import snpio  # type: ignore

    if not hasattr(snpio, "SNPioMultiQC"):

        class _DummyMQC:
            @staticmethod
            def queue_html(*args, **kwargs):
                return None

            @staticmethod
            def queue_linegraph(*args, **kwargs):
                return None

            @staticmethod
            def queue_table(*args, **kwargs):
                return None

            @staticmethod
            def queue_heatmap(*args, **kwargs):
                return None

            @staticmethod
            def build(*args, **kwargs):
                return None

        snpio.SNPioMultiQC = _DummyMQC  # type: ignore[attr-defined]
except Exception:
    # Let importorskip handle truly missing snpio installations.
    snpio = None  # type: ignore


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
    from snpio import GenotypeEncoder, VCFReader

    workdir = tmp_path_factory.mktemp("pgsui_runs")
    gd = VCFReader(
        filename=str(example_vcf_path),
        popmapfile=str(example_popmap_path),
        prefix=str(workdir / "example"),
        force_popmap=True,
        plot_format="pdf",
    )

    # Ensure ref/alt alleles are populated for downstream decoding.
    alleles = gd.get_ref_alt_alleles(gd.snp_data)

    gd.ref = alleles[0].tolist() if isinstance(alleles[0], np.ndarray) else alleles[0]
    alts = [x for i, x in enumerate(alleles) if i > 0]

    if isinstance(alts, np.ndarray):
        gd.alt = alts.tolist()
    else:
        gd.alt = alts

    # Patch GenotypeEncoder to carry ref/alt into the encoder instance and
    # avoid file-writing paths that expect VCF/PHYLIP context.
    orig_init = GenotypeEncoder.__init__
    orig_decode = GenotypeEncoder.decode_012

    def _patched_init(self, genotype_data):
        orig_init(self, genotype_data)
        if not hasattr(self, "_ref") or self._ref is None or len(getattr(self, "_ref", [])) == 0:  # type: ignore[attr-defined]
            alleles = genotype_data.get_ref_alt_alleles(genotype_data.snp_data)

            if isinstance(alleles[0], np.ndarray):
                self._ref = alleles[0].tolist()  # type: ignore[attr-defined]
            else:
                self._ref = alleles[0]  # type: ignore[attr-defined]

            alts = [x for i, x in enumerate(alleles) if i > 0]
            self._alt = alts.tolist() if isinstance(alts, np.ndarray) else alts  # type: ignore[attr-defined]

        # Force a known filetype branch in decode_012.
        self.filetype = "vcf"

    def _patched_decode(self, X, write_output: bool = True, is_nuc: bool = False):  # type: ignore[override]
        # Bypass file writing and use a simple safe mapping for test assertions.
        arr = pytest.importorskip("numpy").asarray(X)
        mapper = {
            0: "A",
            1: "C",
            2: "G",
            -9: "A",
            "0": "A",
            "1": "C",
            "2": "G",
            "-9": "A",
        }
        flat = [
            mapper.get(int(x), "A") if not isinstance(x, str) else mapper.get(x, "A")
            for x in arr.ravel()
        ]
        return pytest.importorskip("numpy").array(flat, dtype="<U1").reshape(arr.shape)

    GenotypeEncoder.__init__ = _patched_init  # type: ignore[assignment]
    GenotypeEncoder.decode_012 = _patched_decode  # type: ignore[assignment]

    return gd
