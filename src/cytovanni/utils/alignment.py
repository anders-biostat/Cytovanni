"""Batch mixing of paired samples: k-nearest-neighbour counts and the Seurat alignment score (SAS).

Two aliquots of the same sample measured in different batches should look alike after a batch correction, so their
cells should be well mixed in marker space. The Seurat alignment score (Butler et al. 2018) measures this mixing:
both batches are subsampled to the same number of cells, and for every cell the number ``x`` of its k nearest
neighbours that come from its own batch is counted. With the mean ``x_bar``,

    SAS = 1 - (x_bar - k/2) / (k/2) = 2 (1 - x_bar / k),

clipped at 1. Perfect mixing gives ``x_bar = k/2`` and SAS = 1 (sampling noise pushes it slightly above 1 before the
clip), fully separated batches give ``x_bar = k`` and SAS = 0. Restricting the cells to one cell type (the neighbours
are then found among that cell type only) gives a score per cell type.

Workflow
--------
1. `alignment_design`: balanced, seeded subsamples for the whole sample and every cell type, with a minimum-cell rule.
2. `alignment_scores`: the SAS of every group of a design for one marker matrix (reuse the design for several
   versions of the data, e.g. several batch corrections, so that all of them see the same cells).
   `pair_alignment_scores` combines both for two samples given as DataFrames.
3. Optional: `weighted_celltype_score` summarises the per-cell-type scores into one number per pair.

kNN backends
------------
`knn_same_batch_counts` does an exact Euclidean neighbour search with scikit-learn (CPU, default) or torch (CPU or
GPU). torch is only imported when used. Every torch path takes a ``device`` ("auto", "cpu", "cuda" or e.g.
"cuda:1"); without a usable GPU it falls back to the CPU with a `MissingGPUWarning`, and it warns with a
`CytovanniWarning` when a search is large enough to be much slower on the CPU than on a GPU. The backends agree up to
the order of tied or numerically almost tied neighbours (float32 distances), which changes the SAS by far less than
its subsampling noise (< 1e-3 on our data).
"""
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from ..exceptions import CytovanniWarning, MissingGPUWarning

# A torch search on the CPU is warned about above this many distance evaluations (query cells x reference cells).
# Measured (float32, 30 markers, k=50; A40 vs. 4 CPU cores): the GPU is ~30x faster than torch or sklearn on the CPU for
# every search of >= 10,000 cells, but below ~2e9 distances (~45,000 cells) the CPU still needs only a few seconds, so
# only larger searches warn (60,000 cells: 11 s CPU vs. 0.3 s GPU).
CPU_SLOW_DISTANCES = 2e9


# ================================================================ helpers
def stable_seed(*parts):
    """Integer seed derived from any JSON-serialisable parts, stable across runs, machines and Python versions.

    Unlike ``hash()``, the result does not depend on the interpreter's hash randomisation. Used to give every
    (sample pair, annotation level, cell type) group of `alignment_design` its own reproducible subsample, and
    convenient for `cytovanni.utils.batcheval.gaussian_negative` seeds per sample.

    Parameters
    ----------
    *parts
        JSON-serialisable values (str, int, float, bool, None, lists of these).

    Returns
    -------
    int
        Seed in [0, 2**32), the first 4 bytes (little endian) of ``sha256(json.dumps(list(parts)))``.

    Examples
    --------
    >>> stable_seed(0, "identity:IS_1002", "annot", "B cells")  # the same number in every session
    """
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "little")


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_device(device="auto"):
    """Torch device to compute on, falling back to the CPU if no GPU is usable.

    Parameters
    ----------
    device : str or torch.device
        ``"auto"`` (CUDA if available, otherwise CPU), ``"cpu"``, ``"cuda"`` or a specific device such as
        ``"cuda:1"``.

    Returns
    -------
    torch.device

    Warns
    -----
    MissingGPUWarning
        A CUDA device was requested explicitly but torch finds none; the CPU is used instead.
    """
    import torch
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("Requested a CUDA device, but torch finds no GPU: computing on the CPU instead (much slower for "
                      "large searches).", MissingGPUWarning, stacklevel=3)
        return torch.device("cpu")
    return device


def _warn_if_slow(device, n_query, n_ref, what):
    """CytovanniWarning if a brute-force search of this size runs on the CPU (see CPU_SLOW_DISTANCES)."""
    if device.type == "cpu" and float(n_query) * float(n_ref) > CPU_SLOW_DISTANCES:
        warnings.warn(f"{what}: {n_query} x {n_ref} distances on the CPU; on a GPU (device='cuda') this is typically "
                      f">20x faster.", CytovanniWarning, stacklevel=3)


class _HighestMatmulPrecision:
    """Disable TF32 and use full float32 matmuls inside the block, restoring torch's settings afterwards.

    TF32 (the default on Ampere GPUs for some operations) rounds the inputs of the distance matmul to 10 mantissa bits,
    which reorders close neighbours far more often than float32 rounding does.
    """
    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.tf32 = self.torch.backends.cuda.matmul.allow_tf32
        self.precision = self.torch.get_float32_matmul_precision()
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.torch.set_float32_matmul_precision("highest")

    def __exit__(self, *exc):
        self.torch.backends.cuda.matmul.allow_tf32 = self.tf32
        self.torch.set_float32_matmul_precision(self.precision)


def _batch_codes(batch):
    """Integer codes of arbitrary batch labels (identity for integer arrays)."""
    batch = np.asarray(batch)
    if np.issubdtype(batch.dtype, np.integer):
        return batch
    return np.unique(batch, return_inverse=True)[1].reshape(-1)


# ================================================================ kNN counts
def knn_same_batch_counts(x, batch, k=50, backend="sklearn", device="auto", precision="float32", chunk_size=1024,
                          exclude_self="index"):
    """For every cell, the number of its k nearest neighbours (Euclidean) that have the same batch label.

    Exact brute-force search; the cell itself is never counted as its own neighbour.

    Parameters
    ----------
    x : array-like, shape (events, marker)
        Coordinates, e.g. arcsinh-transformed marker intensities. Must be finite.
    batch : array-like, shape (events,)
        Batch label of every cell (any labels; compared for equality).
    k : int
        Number of neighbours. Reduced to ``events - 1`` for small inputs; with fewer than 2 cells all counts are 0.
    backend : {"sklearn", "torch"}
        ``"sklearn"``: `sklearn.neighbors.NearestNeighbors` (brute force, CPU, in the dtype of `x`).
        ``"torch"``: squared distances ``|a|^2 + |b|^2 - 2 a.b`` with torch on `device`, TF32 disabled. If torch is
        not installed, falls back to sklearn with a `CytovanniWarning`.
    device : str or torch.device
        torch backend only, see `resolve_device`. The sklearn backend always runs on the CPU.
    precision : {"float32", "float64"}
        torch backend only. ``"float64"`` computes the distances in float64 and re-ranks the ``k + 8`` best candidates
        by their exact squared differences (the most accurate ordering, about 2-4x slower on GPUs).
    chunk_size : int
        Number of query cells per step (memory: about ``chunk_size * events`` distances).
    exclude_self : {"index", "first"}
        How the cell itself is removed from its neighbours. ``"index"``: by its index, also when other cells have the
        same coordinates (exact duplicates). ``"first"``: drop the first returned neighbour, which is only correct if
        no cell has duplicates.

    Returns
    -------
    numpy.ndarray of int, shape (events,)
        Same-batch neighbour count of every cell, in [0, k].

    Warns
    -----
    MissingGPUWarning
        torch backend: a CUDA device was requested but is not available (computed on the CPU).
    CytovanniWarning
        torch backend on the CPU for a search with more than `CPU_SLOW_DISTANCES` distances.

    See Also
    --------
    alignment_score_from_counts : turns the counts into the SAS.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(2000, 5)); batch = np.repeat([0, 1], 1000)
    >>> counts = knn_same_batch_counts(x, batch, k=50)
    >>> counts.mean()  # about k/2 = 25 for perfectly mixed batches
    """
    if exclude_self not in ("index", "first"):
        raise ValueError(f"exclude_self must be 'index' or 'first', not {exclude_self!r}")
    if precision not in ("float32", "float64"):
        raise ValueError(f"precision must be 'float32' or 'float64', not {precision!r}")
    drop_first = exclude_self == "first"
    batch = _batch_codes(batch)
    n = len(x)
    k = min(k, n - 1)
    if k < 1:
        return np.zeros(n, dtype=int)
    if backend == "torch":
        torch = _import_torch()
        if torch is None:
            warnings.warn("torch is not installed: using the sklearn backend.", CytovanniWarning, stacklevel=2)
            backend = "sklearn"
        else:
            return _knn_counts_torch(torch, x, batch, k, device, precision, chunk_size, drop_first)
    if backend != "sklearn":
        raise ValueError(f"backend must be 'sklearn' or 'torch', not {backend!r}")
    from sklearn.neighbors import NearestNeighbors
    model = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=1).fit(x)
    results = []
    for start in range(0, n, chunk_size):
        ids = np.asarray(model.kneighbors(x[start:start + chunk_size], return_distance=False))
        if drop_first:
            ids = ids[:, 1:]
        else:
            keep = ids != np.arange(start, start + len(ids))[:, None]
            # with ties the cell itself may be missing from the k+1 candidates: take the first k others either way
            order = np.argsort(~keep, axis=1, kind="stable")[:, :k]
            ids = np.take_along_axis(ids, order, axis=1)
        results.append((batch[ids] == batch[start:start + len(ids), None]).sum(1))
    return np.concatenate(results)


def _knn_counts_torch(torch, x, batch, k, device, precision, chunk_size, drop_first):
    device = resolve_device(device)
    n = len(x)
    _warn_if_slow(device, n, n, "knn_same_batch_counts")
    refined = precision == "float64"
    with _HighestMatmulPrecision(torch), torch.no_grad():
        a = torch.as_tensor(np.asarray(x, dtype=np.float64 if refined else np.float32), device=device)
        b = torch.as_tensor(batch, device=device)
        norm = (a * a).sum(1)
        exact = torch.as_tensor(np.asarray(x, dtype=np.float64), device=device) if refined else None
        results = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dist = norm[start:end, None] + norm[None, :] - 2 * a[start:end] @ a.T
            if not drop_first:
                rows = torch.arange(end - start, device=device)
                dist[rows, rows + start] = torch.inf
            candidates = min(n - 1, k + 8) if refined else k
            ids = dist.topk(candidates + int(drop_first), largest=False, sorted=True).indices[:, int(drop_first):]
            if refined:
                d = ((exact[ids] - exact[start:end, None]) ** 2).sum(2)
                ids = ids.gather(1, d.topk(k, largest=False, sorted=True).indices)
            results.append((b[ids] == b[start:end, None]).sum(1).cpu().numpy())
    return np.concatenate(results)


def knn_same_batch_counts_grouped(groups, k=50, device="auto", precision="float32", max_elements=8_000_000):
    """`knn_same_batch_counts` (torch backend) for many small, independent groups at once.

    Groups of similar size are padded to the next power of two and searched together in one batched matrix product,
    which is much faster on a GPU than one search per group (e.g. the per-cell-type searches of a sample pair).
    Neighbours are only searched within each group. Without torch, every group is searched with the sklearn backend.

    Parameters
    ----------
    groups : sequence of (x, batch)
        Coordinates, shape (events_i, marker), and batch labels, shape (events_i,), of every group (same number of
        markers in all groups).
    k : int
        Number of neighbours; reduced to ``events_i - 1`` per group.
    device : str or torch.device
        See `resolve_device`.
    precision : {"float32", "float64"}
        Dtype of the distance computation (no re-ranking step here).
    max_elements : int
        Maximal number of distance entries per batched product (memory bound).

    Returns
    -------
    list of numpy.ndarray of int
        Same-batch neighbour counts of every group, in the order of `groups`.

    Warns
    -----
    MissingGPUWarning
        A CUDA device was requested but is not available (computed on the CPU).
    CytovanniWarning
        On the CPU, if the total search is larger than `CPU_SLOW_DISTANCES` distances.
    """
    torch = _import_torch()
    if torch is None:
        warnings.warn("torch is not installed: searching every group with the sklearn backend.", CytovanniWarning,
                      stacklevel=2)
        return [knn_same_batch_counts(x, b, k=k) for x, b in groups]
    device = resolve_device(device)
    groups = [(np.asarray(x), _batch_codes(b)) for x, b in groups]
    _warn_if_slow(device, sum(len(x) for x, _ in groups), max((len(x) for x, _ in groups), default=0),
                  "knn_same_batch_counts_grouped")
    buckets, result = {}, {}
    for i, (x, b) in enumerate(groups):
        size = 2 ** int(np.ceil(np.log2(max(2, len(x)))))
        buckets.setdefault(size, []).append((i, x, b))
    dtype = np.float64 if precision == "float64" else np.float32
    with _HighestMatmulPrecision(torch), torch.no_grad():
        for size, values in buckets.items():
            step = max(1, max_elements // (size * size))
            for start in range(0, len(values), step):
                part = values[start:start + step]
                a = np.zeros((len(part), size, part[0][1].shape[1]), dtype=dtype)
                b = np.zeros((len(part), size), dtype=int)
                lengths = []
                for j, (_, x, y) in enumerate(part):
                    a[j, :len(x)], b[j, :len(x)] = x, y
                    lengths.append(len(x))
                at = torch.as_tensor(a, device=device)
                bt = torch.as_tensor(b, device=device)
                norm = (at * at).sum(2)
                d = norm[:, :, None] + norm[:, None, :] - 2 * torch.bmm(at, at.transpose(1, 2))
                d.diagonal(dim1=1, dim2=2).fill_(torch.inf)
                for j, n in enumerate(lengths):
                    d[j, :, n:] = torch.inf  # padding is never a neighbour
                ids = d.topk(min(k, size - 1), largest=False, sorted=True).indices
                hits = bt.gather(1, ids.reshape(len(part), -1)).reshape(ids.shape) == bt[:, :, None]
                for j, (i, x, _) in enumerate(part):
                    result[i] = hits[j, :len(x), :min(k, len(x) - 1)].sum(1).cpu().numpy()
    return [result[i] for i in range(len(groups))]


# ================================================================ SAS
def alignment_score_from_counts(counts, k, clip=True):
    """Seurat alignment score from same-batch neighbour counts of a batch-balanced set of cells.

    ``SAS = 2 (1 - mean(counts) / k)``, i.e. ``1 - (x_bar - k/2) / (k/2)``.

    Parameters
    ----------
    counts : array-like of int
        Same-batch neighbour counts (`knn_same_batch_counts`) of cells with balanced batch labels.
    k : int
        Number of neighbours used for the counts (for small groups: ``min(k, events - 1)``).
    clip : bool
        Clip the score at 1 (values above 1 are sampling noise of well-mixed batches).

    Returns
    -------
    float
        SAS: 1 for perfect mixing, 0 for fully separated batches. NaN for empty counts or ``k == 0``.
    """
    counts = np.asarray(counts)
    if not len(counts) or not k:
        return np.nan
    score = 2 * (1 - np.mean(counts) / k)
    return min(score, 1.) if clip else score


def alignment_design(batch, annotation=None, min_cells=50, min_rule="per_sample", n_max=100_000, seed=0, key=()):
    """Balanced, seeded cell subsets of two batches for the whole sample and every cell type.

    For every group (the whole sample, ``level="total"``, and every label of every annotation level), the cells of
    the larger batch are subsampled without replacement to the number of cells of the smaller one, the combined
    cells are shuffled and cut to `n_max`. Every group draws from its own `numpy.random.RandomState`, seeded with
    ``stable_seed(seed, *key, level, celltype)``, so its subsample does not depend on the other groups and is the
    same for every version of the data that is scored with the design.

    Parameters
    ----------
    batch : array-like, shape (events,)
        Batch label of every cell; exactly two batches. Integer labels 0/1 are used as given; other labels are sorted
        (``numpy.unique``), the first is "left", the second "right".
    annotation : pandas.DataFrame or dict, optional
        Cell type labels per annotation level, aligned with `batch`: DataFrame columns or dict values (level name ->
        labels, shape (events,)). Labels are compared as strings. Without it only the total score is designed.
    min_cells : int
        Minimum number of cells for a cell type to be scored (see `min_rule`). The whole sample only needs cells in
        both batches.
    min_rule : {"per_sample", "total"}
        ``"per_sample"``: both batches need at least `min_cells` cells of the type; ``"total"``: the two batches
        together need `min_cells` (and each at least one).
    n_max : int
        Maximal number of cells per group after balancing.
    seed : int
        Base seed.
    key : tuple
        Extra seed parts identifying the pair (e.g. ``("identity:IS_1002",)``), so different pairs draw different
        subsamples.

    Returns
    -------
    dict
        ``groups``: list of dicts with ``level``, ``celltype`` (``"total"`` for the whole sample), ``indices``
        (selected cells, positions into `batch`), ``mask`` (all cells of the group), ``left_cells``, ``right_cells``
        (cells of the group in each batch before balancing) and ``eligible`` (minimum-cell rule passed);
        ``batch``: integer batch codes (0 = left, 1 = right), shape (events,).
    """
    if min_rule not in ("total", "per_sample"):
        raise ValueError(f"min_rule must be 'total' or 'per_sample', not {min_rule!r}")
    codes = _batch_codes(batch)
    if set(np.unique(codes)) - {0, 1}:
        raise ValueError("alignment_design needs exactly two batches (codes 0 and 1)")
    if annotation is None:
        annotation = {}
    elif isinstance(annotation, pd.DataFrame):
        annotation = {c: annotation[c] for c in annotation.columns}
    levels = [("total", None), *annotation.items()]
    design = []
    for level, labels_all in levels:
        values = None if labels_all is None else np.asarray(labels_all).astype(str)
        if values is not None and len(values) != len(codes):
            raise ValueError(f"annotation level {level!r} has {len(values)} labels for {len(codes)} cells")
        labels = ["total"] if values is None else np.unique(values)
        for ct in labels:
            mask = np.arange(len(codes)) if values is None else np.flatnonzero(values == ct)
            counts = np.bincount(codes[mask], minlength=2)
            eligible = bool(counts.min() > 0 and (values is None or (counts.sum() >= min_cells if min_rule == "total"
                                                                        else counts.min() >= min_cells)))
            rng = np.random.RandomState(stable_seed(seed, *key, level, str(ct)))
            chosen = np.concatenate([rng.choice(mask[codes[mask] == b], counts.min(), replace=False) for b in [0, 1]])
            chosen = rng.permutation(chosen)[:n_max]
            design.append(dict(level=level, celltype=str(ct), indices=chosen, mask=mask, left_cells=int(counts[0]),
                               right_cells=int(counts[1]), eligible=eligible))
    return dict(groups=design, batch=codes)


def _arcsinh(values, cofactor):
    """arcsinh(values / cofactor), infinite values mapped to +-1e13 (keeps the dtype of `values`)."""
    if cofactor is None:
        return np.ascontiguousarray(values)
    return np.ascontiguousarray(np.clip(np.arcsinh(values / cofactor), -1e13, 1e13))


def alignment_scores(x, design, k=50, cofactor=1500, scope="celltype", backend="sklearn", device="auto",
                     precision="float32", chunk_size=1024, group_small=True):
    """Seurat alignment score of every group of an `alignment_design` for one marker matrix.

    Parameters
    ----------
    x : array-like or pandas.DataFrame, shape (events, marker)
        Marker intensities of all cells, in the order of the design's ``batch`` (e.g. the left sample's cells
        followed by the right sample's). Linear scale if `cofactor` is given, else used as coordinates directly.
    design : dict
        Output of `alignment_design`.
    k : int
        Number of neighbours (reduced to ``events - 1`` for small groups).
    cofactor : float or None
        Coordinates are ``arcsinh(x / cofactor)`` (infinite values mapped to +-1e13); None: use `x` as is.
    scope : {"celltype", "global"}
        ``"celltype"``: neighbours of a cell type's cells are searched among that cell type only (separate, balanced
        subsample per type). ``"global"``: every type's score uses the counts of the whole-sample search, restricted to
        its cells (the type's cells in that subsample are not balanced between the batches). In both, types that fail
        the design's minimum-cell rule get NaN.
    backend, device, precision, chunk_size
        See `knn_same_batch_counts`.
    group_small : bool
        torch backend only: search all cell types with at most 2048 selected cells together with
        `knn_same_batch_counts_grouped` (faster on GPUs; float32 distances computed batch-wise, so tied or almost tied
        neighbours may be ordered differently than in single searches).

    Returns
    -------
    pandas.DataFrame
        One row per group: ``level``, ``celltype``, ``left_cells``, ``right_cells`` and ``sas`` (NaN for cell types
        that fail the design's minimum-cell rule).

    Examples
    --------
    >>> design = alignment_design(batch, annotation=obs[["Major Lineage"]], min_cells=50)
    >>> scores = alignment_scores(values, design)   # values: linear marker intensities, rows as in batch
    """
    x = _arcsinh(np.asarray(x), cofactor)
    if scope not in ("celltype", "global"):
        raise ValueError(f"scope must be 'celltype' or 'global', not {scope!r}")
    batch, groups = design["batch"], design["groups"]
    rows, small = [], []
    inds = groups[0]["indices"]
    global_counts = knn_same_batch_counts(x[inds], batch[inds], k, backend, device, precision, chunk_size)
    for group in groups:
        item = {key: v for key, v in group.items() if key not in ("indices", "mask", "eligible")}
        item["sas"] = np.nan
        if group["level"] == "total":
            item["sas"] = alignment_score_from_counts(global_counts, min(k, len(inds) - 1))
        elif scope == "global" and group["eligible"]:
            use = np.isin(inds, group["mask"])
            item["sas"] = alignment_score_from_counts(global_counts[use], min(k, len(inds) - 1))
        elif group["eligible"]:
            ix = group["indices"]
            if backend == "torch" and group_small and len(ix) <= 2048:
                small.append((len(rows), x[ix], batch[ix]))
            else:
                hits = knn_same_batch_counts(x[ix], batch[ix], k, backend, device, precision, chunk_size)
                item["sas"] = alignment_score_from_counts(hits, min(k, len(ix) - 1))
        rows.append(item)
    if small:
        counts = knn_same_batch_counts_grouped([(xs, b) for _, xs, b in small], k, device, precision)
        for (i, xs, _), hits in zip(small, counts):
            rows[i]["sas"] = alignment_score_from_counts(hits, min(k, len(xs) - 1))
    return pd.DataFrame(rows)


def pair_alignment_scores(left, right, left_annotation=None, right_annotation=None, key=(), min_cells=50,
                          min_rule="per_sample", n_max=100_000, seed=0, **kwargs):
    """Seurat alignment scores of two samples (e.g. aliquots of one sample measured in two batches).

    Convenience wrapper: `alignment_design` for the concatenated cells (left = batch 0), then `alignment_scores`.
    To score several versions of the same two samples on identical subsamples, build the design once and call
    `alignment_scores` for each.

    Parameters
    ----------
    left, right : pandas.DataFrame, shape (events_i, marker)
        Linear marker intensities of the two samples, same columns in the same order (drop markers that should not
        count, e.g. a live/dead stain, before).
    left_annotation, right_annotation : pandas.DataFrame, optional
        Cell type labels per annotation level (columns), rows aligned with `left` / `right`.
    key, min_cells, min_rule, n_max, seed
        See `alignment_design`.
    **kwargs
        Passed to `alignment_scores` (k, cofactor, scope, backend, device, precision, chunk_size, group_small).

    Returns
    -------
    pandas.DataFrame
        See `alignment_scores`.
    """
    if list(left.columns) != list(right.columns):
        raise ValueError("left and right need the same markers in the same order")
    annotation = None
    if left_annotation is not None:
        annotation = pd.concat([pd.DataFrame(left_annotation), pd.DataFrame(right_annotation)], ignore_index=True)
    batch = np.r_[np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)]
    design = alignment_design(batch, annotation, min_cells=min_cells, min_rule=min_rule, n_max=n_max, seed=seed,
                              key=key)
    cofactor = kwargs.pop("cofactor", 1500)
    x = np.concatenate([_arcsinh(left.to_numpy(), cofactor), _arcsinh(right.to_numpy(), cofactor)])
    return alignment_scores(x, design, cofactor=None, **kwargs)


def weighted_celltype_score(scores, shares, square_root=True, renormalize=True):
    """Weighted mean of per-cell-type scores, one value per row (e.g. per sample pair).

    Weights are proportional to the cell type share (or its square root, which gives rare types more weight than
    their share), normalised per row: ``w_ij = s_ij^p / sum_j s_ij^p`` with ``p = 1/2`` or 1.

    Parameters
    ----------
    scores : pandas.DataFrame, shape (pair, celltype)
        Per-cell-type scores, e.g. SAS; NaN for types that were not scored.
    shares : pandas.DataFrame, shape (pair, celltype)
        Cell type shares per row, e.g. the mean of the two samples' shares. Only these columns are used (drop types
        that should not count before).
    square_root : bool
        Weight by the square root of the shares.
    renormalize : bool
        True: ``sum_j w_ij x_ij / sum_j w_ij`` over the scored (non-NaN) types only. False: ``sum_j w_ij x_ij`` with
        the weights of all types, i.e. unscored types contribute 0.

    Returns
    -------
    pandas.Series
        Weighted score per row; NaN where no type was scored (`renormalize`).
    """
    weights = np.sqrt(shares) if square_root else shares
    weights = weights / weights.to_numpy().sum(1, keepdims=True)
    scores = scores.reindex(index=weights.index, columns=weights.columns)
    numerator = (scores * weights).sum(axis=1)
    if not renormalize:
        return numerator
    denominator = weights.where(scores.notna()).sum(axis=1)
    return numerator / denominator.where(denominator > 0)
