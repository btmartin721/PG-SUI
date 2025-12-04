from __future__ import annotations

from types import SimpleNamespace

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
