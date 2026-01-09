from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from snpio.utils.misc import validate_input_type

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


@pytest.fixture(scope="session", autouse=True)
def _force_plot_show_off_for_tests() -> None:
    """Force PlotConfig.show to False when requested by CI."""
    flag = os.environ.get("PGSUI_TEST_DISABLE_PLOTS", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return

    from pgsui.data_processing.containers import PlotConfig

    orig_init = PlotConfig.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["show"] = False
        orig_init(self, *args, **kwargs)

    PlotConfig.__init__ = _patched_init  # type: ignore[assignment]


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

    # Ensure ref/alt alleles are populated for downstream decoding and aligned.
    snp_arr = np.asarray(gd.snp_data, dtype=object)
    n_loci = snp_arr.shape[1]

    def _coerce_alleles(candidate):
        if candidate is None:
            return None
        arr = np.asarray(candidate, dtype=object)
        if arr.shape[0] == n_loci:
            return arr.tolist()
        if arr.ndim > 1 and arr.shape[-1] == n_loci:
            rows = arr.tolist()
            if len(rows) == 1:
                return list(rows[0])
            return [[row[j] for row in rows] for j in range(n_loci)]
        return None

    alleles = gd.get_ref_alt_alleles(gd.snp_data)

    ref_list = _coerce_alleles(getattr(gd, "ref", None))
    if ref_list is None:
        ref_list = _coerce_alleles(alleles[0])
    if ref_list is None:
        ref_list = (
            alleles[0].tolist() if isinstance(alleles[0], np.ndarray) else list(alleles[0])
        )

    alt_list = _coerce_alleles(getattr(gd, "alt", None))
    if alt_list is None:
        alt_list = _coerce_alleles([x for i, x in enumerate(alleles) if i > 0])
    if alt_list is None:
        alt_list = ["." for _ in range(n_loci)]

    valid_bases = {"A", "C", "G", "T"}
    missing_codes = {"", ".", "N", "NONE", "-", "?", "./.", ".|."}

    def _normalize_token(value):
        if value is None:
            return None
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="ignore")
        s = str(value).upper().strip()
        if not s or s in missing_codes:
            return None
        return s

    def _clean_alt(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            cleaned = []
            for item in value:
                tok = _normalize_token(item)
                if tok in valid_bases:
                    cleaned.append(tok)
            return cleaned
        tok = _normalize_token(value)
        return tok if tok in valid_bases else None

    keep_idx = []
    cleaned_ref = []
    cleaned_alt = []
    for i, (ref_val, alt_val) in enumerate(zip(ref_list, alt_list)):
        ref_tok = _normalize_token(ref_val)
        if ref_tok not in valid_bases:
            continue

        alt_clean = _clean_alt(alt_val)
        if isinstance(alt_clean, list):
            alt_out = alt_clean if len(alt_clean) > 1 else (alt_clean[0] if alt_clean else ".")
        else:
            alt_out = alt_clean if alt_clean is not None else "."

        keep_idx.append(i)
        cleaned_ref.append(ref_tok)
        cleaned_alt.append(alt_out)

    if keep_idx and len(keep_idx) != n_loci:
        snp_arr = snp_arr[:, keep_idx]

    gd.snp_data = snp_arr
    gd.ref = cleaned_ref
    gd.alt = cleaned_alt

    for attr in ("n_loci", "n_snps", "n_sites", "num_loci"):
        if hasattr(gd, attr):
            setattr(gd, attr, snp_arr.shape[1])

    # Patch GenotypeEncoder to carry ref/alt into the encoder instance and
    # avoid file-writing paths that expect VCF/PHYLIP context.
    orig_init = GenotypeEncoder.__init__

    def _patched_init(self, genotype_data):
        orig_init(self, genotype_data)
        if not hasattr(self, "_ref") or self._ref is None or len(getattr(self, "_ref", [])) == 0:  # type: ignore[attr-defined]
            ref_candidate = getattr(genotype_data, "ref", None)
            alt_candidate = getattr(genotype_data, "alt", None)
            n_loci = None
            if getattr(genotype_data, "snp_data", None) is not None:
                n_loci = np.asarray(genotype_data.snp_data).shape[1]

            if ref_candidate is not None and alt_candidate is not None and n_loci:
                if isinstance(ref_candidate, np.ndarray):
                    ref_candidate = ref_candidate.tolist()
                if isinstance(alt_candidate, np.ndarray):
                    alt_candidate = alt_candidate.tolist()
                if len(ref_candidate) == n_loci and len(alt_candidate) == n_loci:
                    self._ref = ref_candidate  # type: ignore[attr-defined]
                    self._alt = alt_candidate  # type: ignore[attr-defined]
                else:
                    ref_candidate = None
                    alt_candidate = None

            if ref_candidate is None or alt_candidate is None:
                alleles = genotype_data.get_ref_alt_alleles(genotype_data.snp_data)

                if isinstance(alleles[0], np.ndarray):
                    self._ref = alleles[0].tolist()  # type: ignore[attr-defined]
                else:
                    self._ref = alleles[0]  # type: ignore[attr-defined]

                alts = [x for i, x in enumerate(alleles) if i > 0]
                self._alt = alts.tolist() if isinstance(alts, np.ndarray) else alts  # type: ignore[attr-defined]

        # Force a known filetype branch in decode_012.
        self.filetype = "vcf"

    def _patched_decode(
        self, X: np.ndarray | pd.DataFrame | List[List[int]], is_nuc: bool = False
    ) -> np.ndarray:
        """Decode 012 or 0-9 integer encodings to single-character IUPAC nucleotides.

        Always returns single-character IUPAC codes:
        - A, C, G, T for homozygotes
        - R, Y, S, W, K, M for heterozygotes
        - N for missing/invalid

        Modes:
        1) Standard 012 (0=REF, 1=HET, 2=ALT, -9/-1=missing) using per-locus REF/ALT from GenotypeData.
        Special case: ALT="." (or empty) is treated as monomorphic and defaults to REF.
        2) IUPAC-integer mode (is_nuc=True) using SNPio's updated order:
        A=0, C=1, G=2, T=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9/-1.

        Args:
            X: Matrix of 012 or 0-9 IUPAC integers.
            is_nuc: If True, interpret inputs as 0-9 IUPAC integers (A=0, C=1, G=2, T=3, ...).

        Returns:
            np.ndarray: Same shape as X, dtype '<U1', with single-character IUPAC codes.

        Raises:
            ValueError: If REF/ALT metadata are unavailable for 012 decoding.
        """
        df = validate_input_type(X, return_type="df")

        if not isinstance(df, pd.DataFrame):
            msg = "Internal error: expected DataFrame after validation."
            self.logger.error(msg)
            raise ValueError(msg)

        # IUPAC ambiguity mapping (unordered pairs → code) for 012→IUPAC.
        pair_to_iupac = {
            frozenset(("A", "G")): "R",
            frozenset(("C", "T")): "Y",
            frozenset(("G", "C")): "S",
            frozenset(("A", "T")): "W",
            frozenset(("G", "T")): "K",
            frozenset(("A", "C")): "M",
        }
        valid_bases = {"A", "C", "G", "T"}

        if is_nuc:
            # UPDATED SNPio order: A=0, C=1, G=2, T=3, W=4, R=5, M=6, K=7, Y=8, S=9, N=-9/-1
            iupac_list = ["A", "C", "G", "T", "W", "R", "M", "K", "Y", "S"]
            mapping = {i: iupac_list[i] for i in range(10)}
            mapping[-9] = "N"
            mapping[-1] = "N"

            # accept strings too
            mapping.update({str(k): v for k, v in mapping.items()})
            return df.replace(mapping).to_numpy(dtype="<U1")

        # ---- Standard 012 decoding using REF/ALT per column ----
        ref_alleles = getattr(self.genotype_data, "ref", None)
        alt_alleles = getattr(self.genotype_data, "alt", None)

        if ref_alleles is None or len(ref_alleles) == 0:
            ref_alleles = getattr(self.genotype_data, "_ref", None)
        if ref_alleles is None or len(ref_alleles) == 0:
            ref_alleles = getattr(self, "_ref", None)

        if alt_alleles is None or len(alt_alleles) == 0:
            alt_alleles = getattr(self.genotype_data, "_alt", None)
        if alt_alleles is None or len(alt_alleles) == 0:
            alt_alleles = getattr(self, "_alt", None)

        if ref_alleles is None or alt_alleles is None:
            msg = (
                "Reference and alternate alleles are not available in GenotypeData; "
                "cannot decode 012 matrix."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        df_out = df.copy().astype(object)
        n_cols = len(df_out.columns)

        if isinstance(ref_alleles, np.ndarray):
            ref_alleles = ref_alleles.tolist()
        if isinstance(alt_alleles, np.ndarray):
            alt_alleles = alt_alleles.tolist()

        if len(ref_alleles) != n_cols or len(alt_alleles) != n_cols:
            try:
                alleles = self.genotype_data.get_ref_alt_alleles(
                    self.genotype_data.snp_data
                )
            except Exception:
                alleles = None

            if alleles:
                ref_candidate = alleles[0]
                if isinstance(ref_candidate, np.ndarray):
                    ref_candidate = ref_candidate.tolist()
                if len(ref_candidate) == n_cols:
                    ref_alleles = ref_candidate

                alt_candidate = [x for i, x in enumerate(alleles) if i > 0]
                if isinstance(alt_candidate, np.ndarray):
                    alt_candidate = alt_candidate.tolist()
                if alt_candidate:
                    if len(alt_candidate) == n_cols:
                        alt_alleles = alt_candidate
                    else:
                        try:
                            if all(len(a) == n_cols for a in alt_candidate):
                                alt_alleles = [
                                    [a[i] for a in alt_candidate] for i in range(n_cols)
                                ]
                        except TypeError:
                            pass

        if len(alt_alleles) != n_cols:
            try:
                if alt_alleles and all(len(a) == n_cols for a in alt_alleles):
                    alt_alleles = [[a[i] for a in alt_alleles] for i in range(n_cols)]
            except TypeError:
                pass

        if len(ref_alleles) != n_cols or len(alt_alleles) != n_cols:
            msg = (
                "Reference and alternate alleles do not align with genotype columns; "
                "cannot decode 012 matrix."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        for j, col in enumerate(df_out.columns):
            ref = ref_alleles[j]
            alt = alt_alleles[j]

            # Normalize REF
            ref = "" if ref is None else str(ref).upper()

            # Normalize ALT:
            #   - If list-like, pick the first non-missing ALT that isn't "."
            #   - Treat "." or "" as "no ALT" (monomorphic) -> default to REF later
            if isinstance(alt, (list, tuple)):
                alt_clean = None
                for a in alt:
                    if a is None:
                        continue
                    s = str(a).upper()
                    if s not in {".", ""}:
                        alt_clean = s
                        break
                alt = alt_clean  # may be None
            else:
                if alt is None:
                    alt = None
                else:
                    s = str(alt).upper()
                    alt = None if s in {".", ""} else s

            ref_is_std = ref in valid_bases

            # ALT="." (or empty/missing) indicates monomorphic site -> treat ALT as REF for decoding.
            monomorphic = alt is None
            if monomorphic:
                alt = ref

            alt_is_std = alt in valid_bases

            if ref_is_std and alt_is_std:
                # If monomorphic, ref==alt and het_code becomes ref (prevents "N" injection)
                het_code = (
                    ref if ref == alt else pair_to_iupac.get(frozenset((ref, alt)), "N")
                )
                col_map = {
                    0: ref,
                    "0": ref,
                    1: het_code,
                    "1": het_code,
                    2: alt,
                    "2": alt,
                    -9: "N",
                    "-9": "N",
                    -1: "N",
                    "-1": "N",
                }
            elif ref_is_std and not alt_is_std:
                # ALT is truly invalid (not "."), cannot decode 1/2 meaningfully
                col_map = {
                    0: ref,
                    "0": ref,
                    1: "N",
                    "1": "N",
                    2: "N",
                    "2": "N",
                    -9: "N",
                    "-9": "N",
                    -1: "N",
                    "-1": "N",
                }
            elif not ref_is_std and alt_is_std:
                col_map = {
                    0: "N",
                    "0": "N",
                    1: "N",
                    "1": "N",
                    2: alt,
                    "2": alt,
                    -9: "N",
                    "-9": "N",
                    -1: "N",
                    "-1": "N",
                }
            else:
                col_map = {
                    0: "N",
                    "0": "N",
                    1: "N",
                    "1": "N",
                    2: "N",
                    "2": "N",
                    -9: "N",
                    "-9": "N",
                    -1: "N",
                    "-1": "N",
                }

            df_out[col] = df_out[col].map(col_map)

        return df_out.to_numpy(dtype="<U1")

    GenotypeEncoder.__init__ = _patched_init  # type: ignore[assignment]
    GenotypeEncoder.decode_012 = _patched_decode  # type: ignore[assignment]

    return gd
