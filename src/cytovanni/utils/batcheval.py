"""Evaluation of batch corrections with replicate pairs: per-cell-type summaries, paired CV tables, counting-noise
corrections, a Gaussian-negative control and replicate rank correlations.

A replicate pair is two aliquots of the same sample measured in different batches ("left" and "right"). After a good
batch correction, every per-sample summary (cell type proportions, mean marker intensities (MFI) and population widths
(sigmaFI) per cell type) should agree within each pair. This module turns per-cell data into such summaries and the
summaries into CV tables over markers x cell types, using the CVs of `cytovanni.utils.evl` (fractions by default).

Workflow
--------
1. Per sample: `celltype_counts` (counts and shares per annotation level) and `celltype_marker_summary` (mean /
   median / standard deviation of ``arcsinh(x / cofactor)`` per cell type). `wide_by_marker` reshapes the summaries
   of many samples into one (sample, celltype) table per marker.
2. Over replicate pairs (or a set of repeated measurements of one sample, ``samples=``): `proportion_cv_table`,
   `mfi_cv_table`, `width_cv_table`, `replicate_correlation_table`.
3. Optional counting-noise correction: `proportion_cv_counting_noise` / `mfi_cv_counting_noise` simulate the CVs that
   finite cell numbers alone produce, `subtract_counting_noise` removes them in quadrature.
4. Controls: `gaussian_negative` replaces low intensities by Gaussian noise, so that a mixing score (e.g. the SAS of
   `cytovanni.utils.alignment`) no longer rewards agreement of the noise around zero.

Pairs are given as a DataFrame with columns ``left`` and ``right`` (sample names, the index of the summary tables), or
as a sequence of (left, right) tuples. Random functions take ``rng``: None (numpy's global legacy state, as
``np.random.normal``), an int (seed of a new `numpy.random.RandomState`; reproduces ``np.random.seed(int)``), a
`numpy.random.RandomState` or a `numpy.random.Generator`.
"""
import warnings

import numpy as np
import pandas as pd

from ..exceptions import CytovanniWarning
from .evl import (get_asinh_paired_CV, get_asinh_paired_CV_alongdiag, get_asinh_single_CV, get_positive_paired_CV,
                  get_positive_paired_CV_alongdiag, get_positive_single_CV, symmetric_correlation)
from .alignment import stable_seed  # noqa: F401  (re-exported: seeds for gaussian_negative)

QUANTILES = (15.87, 50, 84.13)  # median and +-1 sigma of a normal distribution, in percent


# ================================================================ helpers
def _rng(rng):
    """None -> numpy's global legacy state; int -> new RandomState(int); RandomState / Generator as given."""
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, (int, np.integer)):
        return np.random.RandomState(int(rng))
    return rng


def _pairs(pairs):
    """(left, right) sample arrays of a pair table or a sequence of tuples."""
    if isinstance(pairs, pd.DataFrame):
        return pairs["left"].to_numpy(), pairs["right"].to_numpy()
    pairs = list(pairs)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _levels(annotation):
    if isinstance(annotation, pd.DataFrame):
        return {c: annotation[c] for c in annotation.columns}
    if isinstance(annotation, pd.Series):
        return {annotation.name if annotation.name is not None else "celltype": annotation}
    return dict(annotation)


def _check_paired_or_single(pairs, samples):
    if (pairs is None) == (samples is None):
        raise ValueError("give exactly one of pairs (replicate pairs) or samples (repeated measurements)")


class _NaNCounter:
    """Silence the per-value CytovanniWarnings of evl (too few values) and emit one summary warning instead."""
    def __init__(self, fname):
        self.fname = fname

    def __enter__(self):
        self.catcher = warnings.catch_warnings()
        self.catcher.__enter__()
        warnings.simplefilter("ignore", CytovanniWarning)
        return self

    def __exit__(self, *exc):
        self.catcher.__exit__(*exc)

    def report(self, values, total):
        n = int(np.sum(~np.isfinite(np.asarray(values, dtype=float))))
        if n:
            warnings.warn(f"{self.fname}: {n} of {total} CVs are NaN (too few finite values or pairs).",
                          CytovanniWarning, stacklevel=3)


def _drop(frame, exclude_celltypes):
    if len(exclude_celltypes):
        frame = frame[~frame["celltype"].isin(exclude_celltypes)]
    return frame.reset_index(drop=True)


# ================================================================ per-sample summaries
def celltype_counts(annotation, min_share=5e-5):
    """Cell counts and shares of every cell type, for one sample and every annotation level.

    Parameters
    ----------
    annotation : pandas.DataFrame, pandas.Series or dict
        Cell type label of every cell, one column (or dict entry) per annotation level. Labels are compared as
        strings.
    min_share : float
        Shares are clipped below at this value, so that log-scale metrics stay finite (a type with 0 cells in this
        sample gets no row here, but shares of other samples' tables may be filled with `min_share`).

    Returns
    -------
    pandas.DataFrame
        Columns ``level``, ``celltype``, ``count``, ``share`` (count / cells of the sample, clipped at `min_share`),
        sorted by level (input order) and label.
    """
    frames = []
    for level, labels in _levels(annotation).items():
        names, counts = np.unique(np.asarray(pd.Series(labels).astype(str)), return_counts=True)
        frames.append(pd.DataFrame(dict(level=level, celltype=names, count=counts,
                                        share=np.clip(counts / counts.sum(), min_share, None))))
    return pd.concat(frames, ignore_index=True)


def celltype_marker_summary(values, annotation, cofactor=1500, clip=(-12, 12), stats=("mean", "median"),
                            observed=False):
    """Per-cell-type statistics of arcsinh-transformed marker intensities, for one sample and every level.

    ``y = clip(arcsinh(values / cofactor), *clip)``, then grouped by cell type: ``"mean"`` (MFI on the arcsinh
    scale), ``"median"``, ``"std"`` (population width, sigmaFI, ddof=1) or any other pandas groupby reduction.

    Parameters
    ----------
    values : pandas.DataFrame, shape (events, marker)
        Linear marker intensities of one sample.
    annotation : pandas.DataFrame, pandas.Series or dict
        Cell type labels per annotation level, rows aligned with `values` (same index for DataFrames/Series).
    cofactor : float
        Cofactor of the arcsinh transform.
    clip : tuple of float or None
        Range the transformed values are clipped to (limits the influence of extreme events).
    stats : sequence of str
        Statistics to compute.
    observed : bool
        Only relevant for categorical labels: False gives a row (NaN) also for categories without cells.

    Returns
    -------
    pandas.DataFrame
        Columns ``level``, ``stat``, ``celltype`` and one column per marker.

    Notes
    -----
    For float32 data the mean is computed with ``DataFrame.mean`` per group, which can differ from
    ``groupby().mean()`` by one unit in the last place; this keeps the values identical to our earlier evaluations.
    """
    x = np.arcsinh(values / cofactor)
    if clip is not None:
        x = np.clip(x, *clip)
    frames = []
    for level, labels in _levels(annotation).items():
        grouped = x.groupby(labels, observed=observed)
        for stat in stats:
            df = grouped.apply(lambda g: g.mean()) if stat == "mean" and all(x.dtypes == np.float32) \
                else getattr(grouped, stat)()
            df.index.name = "celltype"
            df = df.reset_index()
            df.insert(0, "stat", stat)
            df.insert(0, "level", level)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def wide_by_marker(frame, sample_key="sample", celltype_key="celltype", markers=None):
    """Reshape a long per-sample summary into one (sample, celltype) table per marker.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long table with one row per (sample, cell type), e.g. `celltype_marker_summary` outputs of several samples
        concatenated with a sample column, restricted to one level and statistic.
    sample_key, celltype_key : str
        Columns holding the sample and cell type names.
    markers : list of str, optional
        Marker columns; default: all columns except the key columns and ``level``, ``stat``, ``panel``.

    Returns
    -------
    dict of str -> pandas.DataFrame
        marker -> table with samples as index and cell types as columns (the input of `mfi_cv_table`,
        `width_cv_table`, `mfi_cv_counting_noise`).
    """
    if markers is None:
        markers = [c for c in frame.columns if c not in (sample_key, celltype_key, "level", "stat", "panel")]
    return {m: frame.pivot(index=sample_key, columns=celltype_key, values=m).rename_axis(index=None, columns=None)
            for m in markers}


# ================================================================ CV tables
def mfi_cv_table(means, pairs=None, samples=None, cofactor=1500, drop_unpaired=True, exclude_markers=(),
                 exclude_celltypes=(), percent=False):
    """CVs of mean marker intensities (MFI CV) per marker and cell type, over replicate pairs or repeated samples.

    For every marker m and cell type c, the arcsinh-scale means of the pairs, ``x_i`` (left) and ``y_i`` (right),
    give ``CV = get_asinh_paired_CV(x, y, cofactor)``: the variance of the symmetrised pairs orthogonal to the
    diagonal, ``var_perp = u^T Sigma u`` with ``u = [-1, 1] / sqrt(2)``, translated to the linear scale at the pooled
    mean ``z0 = cofactor * sinh(mean)`` as ``sqrt(z0^2 + cofactor^2) sqrt(var_perp) / max(|z0|, cofactor)``.
    ``CV_diag`` (`get_asinh_paired_CV_alongdiag`) is the CV of the underlying values without the measurement error,
    from ``(var_along - var_perp) / 2``; ``CV_ratio = CV_diag / CV`` is a signal-to-noise ratio.
    With `samples` instead of pairs: ``CV = get_asinh_single_CV`` over the samples.

    Parameters
    ----------
    means : dict of str -> pandas.DataFrame
        marker -> per-cell-type means on the arcsinh scale, samples as index, cell types as columns (see
        `wide_by_marker`).
    pairs : pandas.DataFrame or sequence of (str, str), optional
        Replicate pairs (columns ``left``, ``right``).
    samples : sequence of str, optional
        Repeated measurements of one sample (e.g. a reference sample measured in every batch), instead of `pairs`.
    cofactor : float
        Cofactor of the arcsinh scale of `means`.
    drop_unpaired : bool
        Pairs only: use only pairs with both values finite, also for ``mean`` (False: ``mean`` over all finite values,
        the CVs always use complete pairs).
    exclude_markers : sequence of str
        Markers to skip (e.g. a live/dead stain).
    exclude_celltypes : sequence of str
        Cell types to drop from the result (e.g. debris or unassigned labels).
    percent : bool
        Report CVs in percent instead of fractions.

    Returns
    -------
    pandas.DataFrame
        One row per marker and cell type (marker-major order): ``marker``, ``celltype``, ``CV``, ``mean`` (pooled
        arcsinh mean) and, for pairs, ``CV_diag``, ``CV_ratio``.

    Warns
    -----
    CytovanniWarning
        Once, with the number of NaN CVs (too few finite values or pairs).

    Examples
    --------
    >>> table = mfi_cv_table(wide_by_marker(summary_mean), pairs, exclude_markers=["livedead"])
    >>> table.groupby("marker").CV.median()
    """
    _check_paired_or_single(pairs, samples)
    scale = 100 if percent else 1
    full = []
    with _NaNCounter("mfi_cv_table") as counter:
        for m, df in means.items():
            if m in exclude_markers:
                continue
            res = []
            for c in df.columns:
                if pairs is not None:
                    left, right = _pairs(pairs)
                    x = df.loc[left, c].to_numpy().flatten()
                    y = df.loc[right, c].to_numpy().flatten()
                    if drop_unpaired:
                        both = np.isfinite(x) & np.isfinite(y)
                        x, y = x[both], y[both]
                    res.append((get_asinh_paired_CV(x, y, cofactor) * scale, np.nanmean(np.hstack([x, y])),
                                get_asinh_paired_CV_alongdiag(x, y, cofactor) * scale))
                else:
                    x = df.loc[list(samples), c].to_numpy().flatten()
                    res.append((get_asinh_single_CV(x, cofactor) * scale, np.nanmean(x)))
            cols = ["CV", "mean", "CV_diag"] if pairs is not None else ["CV", "mean"]
            part = pd.DataFrame(res, index=df.columns, columns=cols)
            if pairs is not None:
                part["CV_ratio"] = part["CV_diag"] / part["CV"]
            part.insert(0, "celltype", part.index)
            part.insert(0, "marker", m)
            full.append(part)
        result = _drop(pd.concat(full).reset_index(drop=True), exclude_celltypes)
    counter.report(result["CV"], len(result))
    return result


def _mean_pairwise_abs(x, log=False):
    x = np.asarray(x)
    x = x[x == x]
    d = np.log(x[:, None] / x[None, :]) if log else x[:, None] - x[None, :]
    return np.mean(np.abs(d)[np.triu_indices(len(x), 1)])


def width_cv_table(means, widths, pairs=None, samples=None, drop_unpaired=True, exclude_markers=(),
                   exclude_celltypes=(), percent=False):
    """CVs of population widths (sigmaFI CV) per marker and cell type, over replicate pairs or repeated samples.

    The widths (standard deviations of the arcsinh-transformed intensities) are strictly positive, so the CVs are
    those of `get_positive_paired_CV` (log scale, variance orthogonal to the diagonal, ``CV = sqrt(exp(var) - 1)``)
    and `get_positive_paired_CV_alongdiag`; with `samples`, `get_positive_single_CV`.

    Parameters
    ----------
    means : dict of str -> pandas.DataFrame
        marker -> per-cell-type means (arcsinh scale), samples x cell types; only used for the ``mean`` column (the
        x position in plots), may come from a different data version than `widths`.
    widths : dict of str -> pandas.DataFrame
        marker -> per-cell-type standard deviations (arcsinh scale), samples x cell types. Defines the markers.
    pairs, samples
        See `mfi_cv_table`.
    drop_unpaired : bool
        Pairs only: ``mean`` over the pairs where both means are finite (False: over all left and right values).
    exclude_markers, exclude_celltypes, percent
        See `mfi_cv_table`.

    Returns
    -------
    pandas.DataFrame
        One row per marker and cell type: ``marker``, ``celltype``, ``mean``, ``mean_abs_diff`` (mean absolute
        difference of the widths within pairs, or between all samples), ``mean_abs_log_ratio`` (same for
        ``|log(w1 / w2)|``), ``CV`` and, for pairs, ``CV_diag``, ``CV_ratio``.
    """
    _check_paired_or_single(pairs, samples)
    scale = 100 if percent else 1
    full = []
    with _NaNCounter("width_cv_table") as counter:
        for m, df in widths.items():
            if m in exclude_markers:
                continue
            if pairs is not None:
                left, right = _pairs(pairs)
                idl, idr = np.hstack([left, right]), np.hstack([right, left])
                a, b = df.loc[idl].reset_index(drop=True), df.loc[idr].reset_index(drop=True)
                diffmean = (a - b).abs().mean()
                logratio = np.log(a / b).abs().mean()
                cv = pd.Series([get_positive_paired_CV(df.loc[left, g], df.loc[right, g]) * scale
                                for g in diffmean.index], index=diffmean.index)
                cv_diag = pd.Series([get_positive_paired_CV_alongdiag(df.loc[left, g], df.loc[right, g]) * scale
                                     for g in diffmean.index], index=diffmean.index)
                mean_table = means[m]
                if drop_unpaired:
                    ma = mean_table.loc[idl].reset_index(drop=True)
                    mb = mean_table.loc[idr].reset_index(drop=True)
                    mean_expr = ma.where(ma.notna() & mb.notna()).mean()
                else:
                    mean_expr = mean_table.loc[idl].mean()
                part = pd.DataFrame([mean_expr, diffmean, logratio, cv, cv_diag, cv_diag / cv],
                                    index=["mean", "mean_abs_diff", "mean_abs_log_ratio", "CV", "CV_diag",
                                           "CV_ratio"]).T
            else:
                samples_ = list(samples)
                mean_expr = means[m].loc[samples_].mean()
                diffmean = pd.Series([_mean_pairwise_abs(df.loc[samples_, g]) for g in df.columns], index=df.columns)
                logratio = pd.Series([_mean_pairwise_abs(df.loc[samples_, g], log=True) for g in df.columns],
                                     index=df.columns)
                cv = pd.Series([get_positive_single_CV(df.loc[samples_, g]) * scale for g in df.columns],
                               index=df.columns)
                part = pd.DataFrame([mean_expr, diffmean, logratio, cv],
                                    index=["mean", "mean_abs_diff", "mean_abs_log_ratio", "CV"]).T
            part.insert(0, "celltype", part.index)
            part.insert(0, "marker", m)
            full.append(part)
        result = _drop(pd.concat(full).reset_index(drop=True), exclude_celltypes)
    counter.report(result["CV"], len(result))
    return result


def _proportion_cv(shares, left, right):
    return [get_positive_paired_CV(shares[c].loc[left], shares[c].loc[right]) for c in shares.columns]


def proportion_cv_table(shares, pairs=None, samples=None, exclude_celltypes=(), percent=False):
    """CVs of cell type proportions over replicate pairs (or repeated samples), one row per cell type.

    Proportions are treated as log-normal: ``CV = sqrt(exp(var_perp) - 1)`` with the variance of the symmetrised log
    shares orthogonal to the diagonal (`get_positive_paired_CV`); ``CV_diag`` from the variance along the diagonal
    (`get_positive_paired_CV_alongdiag`), ``CV_ratio = CV_diag / CV``. With `samples`: `get_positive_single_CV`.

    Parameters
    ----------
    shares : pandas.DataFrame, shape (sample, celltype)
        Cell type shares per sample (> 0; e.g. clipped at 5e-5 as in `celltype_counts`).
    pairs, samples
        See `mfi_cv_table`.
    exclude_celltypes : sequence of str
        Cell types to drop from the result.
    percent : bool
        CVs in percent instead of fractions.

    Returns
    -------
    pandas.DataFrame
        One row per cell type: ``celltype``, ``mean`` (geometric mean share over all samples of the pairs, a sample
        counted once per pair it is in), ``log_std`` (standard deviation of the log shares; pairs: all left and
        right values, ddof=0; samples: ddof=1), ``CV`` and, for pairs, ``r_sym`` (`symmetric_correlation` of the log
        shares), ``CV_diag``, ``CV_ratio``.
    """
    _check_paired_or_single(pairs, samples)
    scale = 100 if percent else 1
    with _NaNCounter("proportion_cv_table") as counter:
        if pairs is not None:
            left, right = _pairs(pairs)
            allind = np.hstack([left, right])
            df = pd.DataFrame([np.exp(np.log(shares.loc[allind]).mean())], index=["mean"]).T
            df["log_std"] = [np.hstack([np.log(shares[c].loc[left]), np.log(shares[c].loc[right])]).std()
                             for c in df.index]
            df["r_sym"] = [symmetric_correlation(np.log(shares[c].loc[left]), np.log(shares[c].loc[right]))
                           for c in df.index]
            df["CV"] = np.array(_proportion_cv(shares[df.index], left, right)) * scale
            df["CV_diag"] = [get_positive_paired_CV_alongdiag(shares[c].loc[left], shares[c].loc[right]) * scale
                             for c in df.index]
            df["CV_ratio"] = df["CV_diag"] / df["CV"]
        else:
            samples_ = list(samples)
            df = pd.DataFrame([np.exp(np.log(shares.loc[samples_]).mean())], index=["mean"]).T
            df["log_std"] = [np.log(shares[c].loc[samples_]).std() for c in df.index]
            df["CV"] = [get_positive_single_CV(shares[c].loc[samples_]) * scale for c in df.index]
        df.insert(0, "celltype", df.index)
        result = _drop(df, exclude_celltypes)
    counter.report(result["CV"], len(result))
    return result


def replicate_correlation_table(values, pairs, celltypes=None):
    """Correlation of one per-cell-type quantity between the left and right samples of replicate pairs.

    For every cell type, the values of the pairs (x_i left, y_i right; pairs with a NaN dropped) give the Pearson
    correlation r and the Spearman rank correlation rho, i.e. how well the ordering of the samples by this quantity
    (e.g. a marker's MFI in one cell type) is reproduced across batches.

    Parameters
    ----------
    values : pandas.DataFrame, shape (sample, celltype)
        One quantity per sample and cell type, e.g. ``means[marker]``.
    pairs : pandas.DataFrame or sequence of (str, str)
        Replicate pairs.
    celltypes : sequence of str, optional
        Cell types (columns) to evaluate, in this order; default all.

    Returns
    -------
    pandas.DataFrame
        One row per cell type: ``celltype``, ``pearson_r``, ``pearson_r2``, ``spearman_rho``, ``spearman_rho2``.
    """
    from scipy.stats import spearmanr
    left, right = _pairs(pairs)
    rows = []
    for c in (values.columns if celltypes is None else celltypes):
        x, y = values[c].loc[left].to_numpy(), values[c].loc[right].to_numpy()
        keep = ~np.isnan(x + y)
        r = np.corrcoef(x[keep], y[keep])[0, 1]
        rho = spearmanr(x[keep], y[keep]).statistic
        rows.append((c, r, r ** 2, rho, rho ** 2))
    return pd.DataFrame(rows, columns=["celltype", "pearson_r", "pearson_r2", "spearman_rho", "spearman_rho2"])


# ================================================================ counting-noise corrections
def resample_pair_shares(counts, pairs, min_share=5e-5, rng=None):
    """Redraw the cell type counts of both samples of every pair from one common composition.

    For each pair, the composition of the sample with more cells is taken as the truth, and both samples are redrawn
    from it, ``multinomial(n_sample, shares_larger)``, keeping their cell numbers. CVs computed from the redrawn
    shares measure only the counting (Poisson-like) noise of finite cell numbers.

    Parameters
    ----------
    counts : pandas.DataFrame, shape (sample, celltype)
        Cell counts per sample.
    pairs : pandas.DataFrame or sequence of (str, str)
        Replicate pairs. A sample in several pairs keeps the draw of its last pair.
    min_share : float
        Redrawn shares are clipped below at this value.
    rng : None, int, numpy.random.RandomState or numpy.random.Generator
        Random state, see the module docstring.

    Returns
    -------
    pandas.DataFrame
        Redrawn shares, same shape as `counts` (samples outside the pairs keep their observed shares).
    """
    rng = _rng(rng)
    resampled = counts.copy()
    for left, right in zip(*_pairs(pairs)):
        left_total, right_total = counts.loc[left].sum(), counts.loc[right].sum()
        if left_total <= right_total:
            small, large, n, m = left, right, left_total, right_total
        else:
            small, large, n, m = right, left, right_total, left_total
        shares = counts.loc[large] / counts.loc[large].sum()
        resampled.loc[small] = rng.multinomial(int(n), shares)
        resampled.loc[large] = rng.multinomial(int(m), shares)
    return np.clip(resampled / resampled.to_numpy().sum(1, keepdims=True), a_min=min_share, a_max=None)


def proportion_cv_counting_noise(counts, pairs, n_resamples=500, min_share=5e-5, exclude_celltypes=(),
                                 percent=False, rng=None):
    """Proportion CVs (`proportion_cv_table` ``CV``) that counting noise alone produces, for `n_resamples` redraws.

    Parameters
    ----------
    counts : pandas.DataFrame, shape (sample, celltype)
        Cell counts per sample.
    pairs : pandas.DataFrame or sequence of (str, str)
        Replicate pairs.
    n_resamples : int
        Number of redraws (`resample_pair_shares`).
    min_share, exclude_celltypes, percent
        As in `resample_pair_shares` / `proportion_cv_table` (use the same values as for the observed table).
    rng : None, int, numpy.random.RandomState or numpy.random.Generator
        Random state, see the module docstring.

    Returns
    -------
    pandas.DataFrame
        Shape (celltype, n_resamples), rows in the order of `proportion_cv_table` (``celltype`` index), columns
        ``redraw_0, ...``; input of `subtract_counting_noise`.
    """
    rng = _rng(rng)
    left, right = _pairs(pairs)
    keep = [c for c in counts.columns if c not in set(exclude_celltypes)]
    scale = 100 if percent else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CytovanniWarning)
        noise = [np.array(_proportion_cv(resample_pair_shares(counts, pairs, min_share, rng)[keep], left, right))
                 for _ in range(n_resamples)]
    return pd.DataFrame(np.array(noise).T * scale, index=pd.Index(keep, name="celltype"),
                        columns=[f"redraw_{i}" for i in range(n_resamples)])


def mfi_cv_counting_noise(means, widths, counts, pairs, n_redraws=10, cofactor=1500, drop_unpaired=True,
                          exclude_markers=(), exclude_celltypes=(), percent=False, rng=None):
    """MFI CVs (`mfi_cv_table` ``CV``) that counting noise alone produces, for `n_redraws` redraws.

    The mean of n cells with width sigma has the standard error ``e = sigma / sqrt(n)``. For every pair, marker and
    cell type, both means are redrawn around the value of the side with the smaller error,
    ``x' ~ N(mu, e_x)``, ``y' ~ N(mu, e_y)``, and the CV of the redrawn pairs is computed as in `mfi_cv_table`.

    Parameters
    ----------
    means, widths : dict of str -> pandas.DataFrame
        marker -> per-cell-type means and standard deviations (arcsinh scale), samples x cell types.
    counts : pandas.DataFrame, shape (sample, celltype)
        Cell counts per sample (the n of each mean).
    pairs : pandas.DataFrame or sequence of (str, str)
        Replicate pairs.
    n_redraws : int
        Number of redraws.
    cofactor, drop_unpaired, exclude_markers, exclude_celltypes, percent
        As for the observed `mfi_cv_table` (use the same values).
    rng : None, int, numpy.random.RandomState or numpy.random.Generator
        Random state, see the module docstring.

    Returns
    -------
    pandas.DataFrame
        Shape (row of `mfi_cv_table`, n_redraws), columns ``redraw_0, ...``, row order as in `mfi_cv_table` with the
        same arguments; input of `subtract_counting_noise`.
    """
    rng = _rng(rng)
    left, right = _pairs(pairs)
    scale = 100 if percent else 1
    redraws, keys = [], None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CytovanniWarning)
        for _ in range(n_redraws):
            cvs, rows = [], []
            for m, df in means.items():
                if m in exclude_markers:
                    continue
                err = widths[m] / np.sqrt(counts.loc[widths[m].index, widths[m].columns].to_numpy())
                err = err[df.columns]
                for c in df.columns:
                    x, y = df.loc[left, c].to_numpy().flatten(), df.loc[right, c].to_numpy().flatten()
                    x_err, y_err = err.loc[left, c].to_numpy().flatten(), err.loc[right, c].to_numpy().flatten()
                    mu = x.copy()
                    swap = x_err > y_err
                    mu[swap] = y[swap]
                    x_sim, y_sim = rng.normal(mu, x_err), rng.normal(mu, y_err)
                    if drop_unpaired:
                        both = np.isfinite(x_sim) & np.isfinite(y_sim)
                        x_sim, y_sim = x_sim[both], y_sim[both]
                    cvs.append(get_asinh_paired_CV(x_sim, y_sim, cofactor) * scale)
                    rows.append((m, c))
            redraws.append(cvs)
            keys = rows
    index = pd.MultiIndex.from_tuples(keys, names=["marker", "celltype"])
    noise = pd.DataFrame(np.array(redraws).T, index=index, columns=[f"redraw_{i}" for i in range(n_redraws)])
    if len(exclude_celltypes):
        noise = noise[~noise.index.get_level_values("celltype").isin(exclude_celltypes)]
    return noise


def subtract_counting_noise(table, noise, column="CV", quantiles=QUANTILES):
    """Remove simulated counting noise from observed CVs in quadrature.

    For every row and redraw j: ``CV_corrected_j = sqrt(max(CV^2 - CV_noise_j^2, 0))``; the redraws are summarised by
    `quantiles` (default: median and the 1-sigma band).

    Parameters
    ----------
    table : pandas.DataFrame
        Observed CV table, e.g. `mfi_cv_table` or `proportion_cv_table`.
    noise : array-like or pandas.DataFrame, shape (len(table), n_redraws)
        Simulated CVs (`mfi_cv_counting_noise`, `proportion_cv_counting_noise`), rows in the order of `table`.
    column : str
        Column of `table` with the observed CV (same units as `noise`).
    quantiles : tuple of 3 floats
        Percentiles (0-100) reported as ``_minus``, central and ``_plus`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with ``CV_noise_rms`` (root mean square of the noise CVs), ``CV_noise_minus``, ``CV_noise``,
        ``CV_noise_plus`` and ``CV_corrected_minus``, ``CV_corrected``, ``CV_corrected_plus``.
    """
    noise = np.asarray(noise)
    if len(noise) != len(table):
        raise ValueError(f"noise has {len(noise)} rows, table {len(table)}")
    observed = table[column].to_numpy()[:, None]
    corrected = np.sqrt(np.clip(observed ** 2 - noise ** 2, 0, None))
    table = table.copy()
    table["CV_noise_rms"] = np.sqrt((noise ** 2).mean(axis=1))
    table["CV_noise_minus"], table["CV_noise"], table["CV_noise_plus"] = np.percentile(noise, quantiles, axis=1)
    table["CV_corrected_minus"], table["CV_corrected"], table["CV_corrected_plus"] = np.percentile(corrected, quantiles,
                                                                                                   axis=1)
    return table


# ================================================================ controls
def gaussian_negative(values, cutoff=1000, scale=200, rng=None):
    """Replace all intensities below `cutoff` by Gaussian noise ``N(0, scale^2)`` (negative control for mixing scores).

    A mixing score such as the SAS also rewards batches whose noise around zero (non-expressed markers) looks alike.
    Replacing every low value by the same Gaussian in all batches removes this part, so the score of the result only
    reflects the expressed (positive) populations. One noise value is drawn for every entry of `values` (whole matrix,
    in C order) before masking, so the draw does not depend on which entries are replaced.

    Parameters
    ----------
    values : pandas.DataFrame or numpy.ndarray, shape (events, marker)
        Linear marker intensities.
    cutoff : float
        Entries ``< cutoff`` are replaced (also large negative values).
    scale : float
        Standard deviation of the noise (e.g. 200, or 25 for a narrow variant).
    rng : None, int, numpy.random.RandomState or numpy.random.Generator
        Random state, see the module docstring. For a reproducible draw per sample use e.g.
        ``rng=stable_seed(0, sample_name)``.

    Returns
    -------
    same type as `values`
        Copy with the replaced entries (a DataFrame keeps index and columns; float32 is upcast to float64). NaN entries
        stay NaN.
    """
    noise = _rng(rng).normal(0, scale, values.shape)
    if isinstance(values, pd.DataFrame):
        return values.mask(values < cutoff, noise)
    values = np.asarray(values)
    return np.where(values < cutoff, noise, values)
