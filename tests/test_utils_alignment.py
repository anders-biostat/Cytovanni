"""Tests of cytovanni.utils.alignment: kNN same-batch counts (sklearn / torch, CPU / CUDA), the Seurat alignment
score (SAS), balanced seeded designs, per-cell-type scores and the device fallbacks."""
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from cytovanni.exceptions import CytovanniWarning, MissingGPUWarning
from cytovanni.utils import alignment
from cytovanni.utils.alignment import (alignment_design, alignment_score_from_counts, alignment_scores,
                                       knn_same_batch_counts, knn_same_batch_counts_grouped, pair_alignment_scores,
                                       resolve_device, stable_seed, weighted_celltype_score)


def reference_counts(x, batch, k):
    """Plain numpy: exact distances, the cell itself removed by index, stable order."""
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(d, np.inf)
    ids = np.argsort(d, axis=1, kind="stable")[:, :k]
    return (batch[ids] == batch[:, None]).sum(1)


@pytest.fixture
def mixed(rng):
    """Two batches from the same distribution (perfect mixing), 600 cells each."""
    return rng.normal(size=(1200, 6)), np.repeat([0, 1], 600)


# ------------------------------------------------------------------ stable_seed
def test_stable_seed_formula_and_determinism():
    expected = int.from_bytes(hashlib.sha256(json.dumps((0, "p", "annot", "B")).encode()).digest()[:4], "little")
    assert stable_seed(0, "p", "annot", "B") == expected
    assert stable_seed(0, "p", "annot", "B") != stable_seed(1, "p", "annot", "B")
    assert 0 <= stable_seed("x") < 2**32


# ------------------------------------------------------------------ kNN counts
def test_knn_counts_sklearn_matches_reference(rng):
    x, batch = rng.normal(size=(300, 4)), rng.integers(0, 2, 300)
    assert np.array_equal(knn_same_batch_counts(x, batch, k=10, chunk_size=64), reference_counts(x, batch, 10))


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_knn_counts_torch_matches_sklearn(rng, device, precision):
    x, batch = rng.normal(size=(800, 5)), rng.integers(0, 2, 800)
    ref = knn_same_batch_counts(x, batch, k=20)
    res = knn_same_batch_counts(x, batch, k=20, backend="torch", device=device, precision=precision, chunk_size=128)
    if precision == "float64":
        assert np.array_equal(res, ref)
    else:  # float32 distances may reorder almost tied neighbours
        assert np.mean(res == ref) > .99
        assert abs(alignment_score_from_counts(res, 20) - alignment_score_from_counts(ref, 20)) < 1e-3


def test_knn_counts_string_labels(rng):
    x = rng.normal(size=(200, 3))
    labels = np.where(rng.random(200) < .5, "batch A", "batch B")
    assert np.array_equal(knn_same_batch_counts(x, labels, k=7), knn_same_batch_counts(x, labels == "batch B", k=7))


@pytest.mark.parametrize("backend", ["sklearn", "torch"])
def test_knn_counts_exclude_self_with_duplicates(rng, backend):
    """Exact duplicates: 'index' removes the cell itself, never its twin."""
    base = rng.normal(size=(100, 3))
    x = np.vstack([base, base])            # every cell has a twin with identical coordinates
    batch = np.r_[np.zeros(100, int), np.ones(100, int)]
    counts = knn_same_batch_counts(x, batch, k=1, backend=backend, device="cpu", precision="float64")
    assert np.array_equal(counts, np.zeros(200, int))  # the only nearest other cell is the twin in the other batch


def test_knn_counts_small_inputs():
    assert np.array_equal(knn_same_batch_counts(np.zeros((1, 2)), [0], k=5), [0])
    assert len(knn_same_batch_counts(np.zeros((0, 2)), np.zeros(0, int), k=5)) == 0
    x = np.array([[0.], [1.], [3.]])
    assert np.array_equal(knn_same_batch_counts(x, [0, 0, 1], k=50), [1, 1, 0])  # k reduced to 2


def test_knn_counts_invalid_arguments(mixed):
    x, batch = mixed
    for kwargs in [dict(backend="faiss"), dict(exclude_self="none"), dict(precision="float16")]:
        with pytest.raises(ValueError):
            knn_same_batch_counts(x, batch, **kwargs)


def test_grouped_matches_single_searches(rng, device):
    groups = [(rng.normal(size=(n, 4)), rng.integers(0, 2, n)) for n in [2, 5, 31, 64, 65, 300]]
    grouped = knn_same_batch_counts_grouped(groups, k=10, device=device, precision="float64")
    for (x, b), res in zip(groups, grouped):
        assert np.array_equal(res, knn_same_batch_counts(x, b, k=10))


# ------------------------------------------------------------------ device handling
def test_resolve_device_auto_and_cpu():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_missing_gpu_falls_back_to_cpu(monkeypatch, mixed):
    x, batch = mixed
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(MissingGPUWarning):
        res = knn_same_batch_counts(x, batch, k=10, backend="torch", device="cuda", precision="float64")
    assert np.array_equal(res, knn_same_batch_counts(x, batch, k=10))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "auto" falls back silently for small problems
        knn_same_batch_counts(x, batch, k=10, backend="torch", device="auto")


def test_large_cpu_search_warns(monkeypatch, mixed):
    x, batch = mixed
    monkeypatch.setattr(alignment, "CPU_SLOW_DISTANCES", 1000)
    with pytest.warns(CytovanniWarning, match="GPU"):
        knn_same_batch_counts(x, batch, k=10, backend="torch", device="cpu")
    with pytest.warns(CytovanniWarning, match="GPU"):
        knn_same_batch_counts_grouped([(x, batch)], k=10, device="cpu")


def test_without_torch_falls_back_to_sklearn(monkeypatch, mixed):
    x, batch = mixed
    monkeypatch.setattr(alignment, "_import_torch", lambda: None)
    with pytest.warns(CytovanniWarning, match="sklearn"):
        res = knn_same_batch_counts(x, batch, k=10, backend="torch")
    assert np.array_equal(res, knn_same_batch_counts(x, batch, k=10))
    with pytest.warns(CytovanniWarning, match="sklearn"):
        res = knn_same_batch_counts_grouped([(x, batch)], k=10)
    assert np.array_equal(res[0], knn_same_batch_counts(x, batch, k=10))


def test_matmul_precision_restored(mixed):
    x, batch = mixed
    before = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    torch.set_float32_matmul_precision("high")
    knn_same_batch_counts(x, batch, k=5, backend="torch", device="cpu")
    assert torch.get_float32_matmul_precision() == "high"
    torch.backends.cuda.matmul.allow_tf32, _ = before
    torch.set_float32_matmul_precision(before[1])


# ------------------------------------------------------------------ SAS
def test_alignment_score_formula():
    assert alignment_score_from_counts([25, 25], 50) == pytest.approx(1.)
    assert alignment_score_from_counts([50, 50], 50) == 0.
    assert alignment_score_from_counts([20, 20], 50) == 1.       # clipped
    assert alignment_score_from_counts([20, 20], 50, clip=False) == pytest.approx(1.2)
    assert alignment_score_from_counts([40], 50) == pytest.approx(1 - (40 - 25) / 25)  # paper formula
    assert np.isnan(alignment_score_from_counts([], 50)) and np.isnan(alignment_score_from_counts([1], 0))


def test_sas_perfect_mixing_and_separation(mixed):
    x, batch = mixed
    counts = knn_same_batch_counts(x, batch, k=50)
    assert alignment_score_from_counts(counts, 50) > .95
    shifted = x + 100 * batch[:, None]
    assert alignment_score_from_counts(knn_same_batch_counts(shifted, batch, k=50), 50) == 0.


# ------------------------------------------------------------------ designs
@pytest.fixture
def two_types(rng):
    """Type A: mixed between batches; type B: shifted in batch 1. Batch 1 has more cells."""
    n0, n1 = 400, 700
    batch = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    labels = np.where(rng.random(n0 + n1) < .5, "A", "B")
    labels[:3] = "C"                                     # a tiny type, only in batch 0
    x = rng.normal(size=(n0 + n1, 4)) + 20 * (labels == "B")[:, None]
    x[(labels == "B") & (batch == 1)] += 30
    return x, batch, pd.DataFrame({"lineage": labels})


def test_design_balanced_reproducible(two_types):
    _, batch, annot = two_types
    d1 = alignment_design(batch, annot, min_cells=20, key=("pair",))
    d2 = alignment_design(batch, annot, min_cells=20, key=("pair",))
    d3 = alignment_design(batch, annot, min_cells=20, key=("other pair",))
    assert [g["celltype"] for g in d1["groups"]] == ["total", "A", "B", "C"]
    for g, g2, g3 in zip(d1["groups"], d2["groups"], d3["groups"]):
        assert np.array_equal(g["indices"], g2["indices"])
        sel = batch[g["indices"]]
        assert (sel == 0).sum() == (sel == 1).sum() == min(g["left_cells"], g["right_cells"])
        assert np.isin(g["indices"], g["mask"]).all() and len(np.unique(g["indices"])) == len(g["indices"])
        if len(g["indices"]) > 10:
            assert not np.array_equal(g["indices"], g3["indices"])
    total = d1["groups"][0]
    assert total["left_cells"] == 400 and total["right_cells"] == 700 and total["eligible"]
    # the RNG of a group depends only on its own key
    rng = np.random.RandomState(stable_seed(0, "pair", "total", "total"))
    first = rng.choice(np.flatnonzero(batch == 0), 400, replace=False)
    assert np.array_equal(np.sort(total["indices"][batch[total["indices"]] == 0]), np.sort(first))


def test_design_min_cell_rules(two_types):
    _, batch, annot = two_types
    n = {g["celltype"]: (g["left_cells"], g["right_cells"]) for g in alignment_design(batch, annot)["groups"]}
    for rule, minimum in [("per_sample", 50), ("total", 50), ("per_sample", 10_000)]:
        groups = alignment_design(batch, annot, min_cells=minimum, min_rule=rule)["groups"]
        for g in groups[1:]:
            l, r = n[g["celltype"]]
            expected = min(l, r) > 0 and (min(l, r) >= minimum if rule == "per_sample" else l + r >= minimum)
            assert g["eligible"] == expected
    assert len(alignment_design(batch, annot, n_max=100)["groups"][0]["indices"]) == 100
    with pytest.raises(ValueError):
        alignment_design(np.r_[batch, 2], None)


def test_design_with_string_batches(two_types):
    _, batch, annot = two_types
    named = np.where(batch == 0, "day 1", "day 2")
    a, b = alignment_design(batch, annot), alignment_design(named, annot)
    assert all(np.array_equal(g["indices"], h["indices"]) for g, h in zip(a["groups"], b["groups"]))


# ------------------------------------------------------------------ per-cell-type scores
@pytest.mark.parametrize("backend", ["sklearn", "torch"])
def test_alignment_scores_per_celltype(two_types, backend, device):
    x, batch, annot = two_types
    design = alignment_design(batch, annot, min_cells=20)
    scores = alignment_scores(x, design, k=30, cofactor=None, backend=backend, device=device).set_index("celltype")
    assert list(scores.columns) == ["level", "left_cells", "right_cells", "sas"]
    assert scores.loc["A", "sas"] > .85 and scores.loc["B", "sas"] == 0.
    assert np.isnan(scores.loc["C", "sas"])                         # only in one batch
    assert 0.2 < scores.loc["total", "sas"] < .8
    glob = alignment_scores(x, design, k=30, cofactor=None, scope="global").set_index("celltype")
    assert glob.loc["total", "sas"] == scores.loc["total", "sas"] and glob.loc["B", "sas"] < .1


def test_alignment_scores_backends_agree(two_types, device):
    x, batch, annot = two_types
    design = alignment_design(batch, annot, min_cells=20)
    ref = alignment_scores(x, design, k=30, cofactor=None)
    for kwargs in [dict(precision="float64", group_small=False), dict(precision="float64"), dict()]:
        res = alignment_scores(x, design, k=30, cofactor=None, backend="torch", device=device, **kwargs)
        assert np.allclose(res.sas, ref.sas, atol=1e-3, equal_nan=True)


def test_alignment_scores_cofactor(two_types):
    x, batch, annot = two_types
    design = alignment_design(batch, annot, min_cells=20)
    linear = np.sinh(x) * 1500
    a, b = alignment_scores(linear, design, k=30), alignment_scores(x, design, k=30, cofactor=None)
    assert np.allclose(a.sas, b.sas, atol=1e-3, equal_nan=True)


def test_pair_alignment_scores(two_types):
    x, batch, annot = two_types
    values = pd.DataFrame(np.sinh(x) * 1500, columns=list("abcd"))
    left, right = values[batch == 0].reset_index(drop=True), values[batch == 1].reset_index(drop=True)
    la, ra = annot[batch == 0].reset_index(drop=True), annot[batch == 1].reset_index(drop=True)
    res = pair_alignment_scores(left, right, la, ra, key=("p",), min_cells=20, k=30)
    design = alignment_design(batch, annot, min_cells=20, key=("p",))
    pd.testing.assert_frame_equal(res, alignment_scores(values, design, k=30))
    with pytest.raises(ValueError):
        pair_alignment_scores(left, right[["b", "a", "c", "d"]])


def test_weighted_celltype_score():
    scores = pd.DataFrame({"A": [1., .5], "B": [0., np.nan]}, index=["p1", "p2"])
    shares = pd.DataFrame({"A": [.64, .64], "B": [.16, .16]}, index=["p1", "p2"])
    w = np.sqrt(np.array([.64, .16])); w = w / w.sum()       # 2/3, 1/3
    res = weighted_celltype_score(scores, shares)
    assert res["p1"] == pytest.approx(w[0]) and res["p2"] == pytest.approx(.5)
    assert weighted_celltype_score(scores, shares, renormalize=False)["p2"] == pytest.approx(.5 * w[0])
    assert weighted_celltype_score(scores, shares, square_root=False)["p1"] == pytest.approx(.8)
