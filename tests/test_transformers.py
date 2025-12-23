from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Skip these tests if snpio isn't installed at all; when present, patch only
# the missing SNPioMultiQC attribute for older releases.
snpio = pytest.importorskip("snpio")  # type: ignore

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

from pgsui.data_processing.transformers import SimMissingTransformer
from snpio import PhylipReader, TreeParser


class _StubRNG:
    """Deterministic choice generator."""

    def __init__(self, seq: list[object]) -> None:
        self.seq = list(seq)
        self.i = 0

    def choice(self, _arr, p=None):
        del p  # unused
        val = self.seq[self.i % len(self.seq)]
        self.i += 1
        return val


class _FakeNode:
    def __init__(self, name: str | None = None, dist: float = 1.0, children=None):
        self.name = name
        self.dist = dist
        self.children = children or []
        self.up = None
        for c in self.children:
            c.up = self

    def is_root(self) -> bool:
        return self.up is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        leaves = []
        for c in self.children:
            leaves.extend(c.get_leaves())
        return leaves


class _FakeTree:
    def __init__(self, root: _FakeNode):
        self.nodes: list[_FakeNode] = []
        self.root = root
        self._assign_indices(root)
        self.nnodes = len(self.nodes)
        self.treenode = self

    def _assign_indices(self, node: _FakeNode) -> None:
        node.idx = len(self.nodes)
        self.nodes.append(node)
        for child in node.children:
            child.up = node
            self._assign_indices(child)

    def traverse(self, order="preorder"):  # noqa: ARG002
        return list(self.nodes)

    def __getitem__(self, idx: int) -> _FakeNode:
        return self.nodes[idx]


class _FakeTreeParser:
    def __init__(self, tree: _FakeTree):
        self.tree = tree


def _make_fake_tree_parser(samples: list[str]) -> _FakeTreeParser:
    leaves = [
        _FakeNode(name=sample, dist=1.0 + (idx * 0.1))
        for idx, sample in enumerate(samples)
    ]
    root = _FakeNode(children=leaves, dist=0.5)
    return _FakeTreeParser(_FakeTree(root))


def _write_newick_tree(path: Path, samples: list[str]) -> None:
    tips = ",".join(f"{sample}:0.1" for sample in samples)
    path.write_text(f"({tips});\n", encoding="utf-8")


def _write_iqtree_qmatrix(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Rate matrix Q",
                "A C G T",
                "A -0.1 0.02 0.03 0.05",
                "C 0.01 -0.09 0.04 0.04",
                "G 0.02 0.02 -0.08 0.04",
                "T 0.03 0.02 0.01 -0.06",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_site_rates(path: Path, count: int) -> None:
    lines = ["Site Rate Cat C_rate"]
    for idx in range(1, count + 1):
        lines.append(f"{idx} 1.0 1 1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sample_clade_resamples_until_overlap() -> None:
    """If a sampled node has no matching tips, keep sampling until one does."""
    bad = _FakeNode(name="bad_tip")
    good = _FakeNode(name="good_tip")
    root = _FakeNode(children=[bad, good])
    tp = _FakeTreeParser(_FakeTree(root))

    gd = SimpleNamespace(samples=["good_tip"])
    transformer = SimMissingTransformer(genotype_data=gd, tree_parser=tp, strategy="nonrandom")

    rng = _StubRNG(seq=[bad.idx, good.idx])  # first hit bad, then good
    tips = transformer._sample_tree(rng=rng)

    assert tips == ["good_tip"]


def test_sample_clade_raises_when_no_matching_tips() -> None:
    bad = _FakeNode(name="bad_tip")
    root = _FakeNode(children=[bad])
    tp = _FakeTreeParser(_FakeTree(root))

    gd = SimpleNamespace(samples=["other"])
    transformer = SimMissingTransformer(genotype_data=gd, tree_parser=tp, strategy="nonrandom")

    with pytest.raises(ValueError):
        transformer._sample_tree()


@pytest.mark.parametrize(
    "strategy",
    [
        "random",
        "random_weighted",
        "random_weighted_inv",
        "nonrandom",
        "nonrandom_weighted",
    ],
)
def test_sim_missing_strategies_run(strategy: str) -> None:
    samples = [f"s{i}" for i in range(6)]
    gd = SimpleNamespace(samples=samples)
    tp = _make_fake_tree_parser(samples) if strategy.startswith("nonrandom") else None

    base = np.array([0, 1, 2, 0, 1, 2], dtype="float32")
    X = np.stack(
        [np.roll(base, shift) for shift in range(12)],
        axis=1,
    ).astype("float32")

    transformer = SimMissingTransformer(
        genotype_data=gd,
        tree_parser=tp,
        prop_missing=0.25,
        strategy=strategy,
        seed=123,
    )
    transformer.fit(X)
    X_masked = transformer.transform(X)

    assert X_masked.shape == X.shape
    assert transformer.sim_missing_mask_.shape == X.shape
    assert int(transformer.sim_missing_mask_.sum()) > 0
    assert not transformer.sim_missing_mask_.all(axis=0).any()


def test_treeparser_file_inputs(tmp_path: Path) -> None:
    phy_path = (
        Path(__file__).resolve().parents[1]
        / "pgsui"
        / "example_data"
        / "phylip_files"
        / "test_n2.phy"
    )
    gd = PhylipReader(filename=str(phy_path), prefix="treeparser-test", verbose=False)

    tree_path = tmp_path / "test.tre"
    qmatrix_path = tmp_path / "test.iqtree"
    siterates_path = tmp_path / "test.rate"

    _write_newick_tree(tree_path, list(gd.samples))
    _write_iqtree_qmatrix(qmatrix_path)
    _write_site_rates(siterates_path, gd.num_snps)

    tp = TreeParser(
        gd,
        treefile=str(tree_path),
        qmatrix=str(qmatrix_path),
        siterates=str(siterates_path),
        verbose=False,
        debug=False,
    )

    tree = tp.tree
    qmat = tp.qmat
    rates = tp.site_rates

    assert tree.ntips == len(gd.samples)
    assert qmat.shape == (4, 4)
    assert list(qmat.columns) == ["A", "C", "G", "T"]
    assert len(rates) == gd.num_snps
