"""Simple per-sample Gaussian mixture fit of rainbow bead peaks, and the likelihood gate based on it.

`fit_rainbow_GMM` assigns every bead of one rainbow bead measurement to a peak (a GMM on the arcsinh-transformed raw
fluorescence channels). Its result (peak labels and peak means) is what `RainbowCalibrator` needs, and
`get_rainbow_GMM_gate` uses the GMM likelihood to remove beads that fit their peak badly (doublets, debris).
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from ..exceptions import GMMAssignmentWarning

def _GMM_choose_asinh_cofactor(adata, percentile=90, target=7, default=1500):
    """Arcsinh cofactor (raw units) that maps the channels' geometric-mean `percentile`-th percentile to `target`.

    Uses the raw fluorescence channels of ``adata.uns['cytoconfig']``; channels with a non-positive percentile are left
    out, and `default` is returned if none is positive.
    """
    percentiles = np.percentile(adata[:,adata.uns["cytoconfig"].channels_fluorescence].layers["raw"], percentile, axis=0)
    if (percentiles>0).sum()>0:
        geomm = np.exp(np.log(percentiles[percentiles>0]).mean())
        return geomm/np.sinh(target)
    return default

def fit_rainbow_GMM(adata, N=None, init_params="kmeans",
                    random_state=None, n_init=10, max_iter=50,
                    clipminperc=.1, clipmaxperc=99.9, warningthreshold=1.5, fct_cofactor=_GMM_choose_asinh_cofactor, Nretry=3, cofactor=None, order_channels=None):
    """Fit a Gaussian mixture to one rainbow bead measurement and assign every bead to a peak.

    The GMM (scikit-learn ``GaussianMixture``, full covariances, one component per peak) is fitted on all fluorescence
    channels of ``adata.uns['cytoconfig']``, on ``arcsinh(raw / cofactor)`` of ``adata.layers['raw']``, after clipping
    every channel to its [`clipminperc`, `clipmaxperc`] percentiles. Run it on scatter-gated beads; the usual workflow
    fits, gates with `get_rainbow_GMM_gate` and fits again on the gated beads.

    Parameters
    ----------
    adata : anndata.AnnData
        One rainbow bead measurement with ``layers['raw']`` and ``uns['cytoconfig']`` (as read by
        `cytovanni.io.readfcs_rainbow_sample`). Changed in place, see Returns.
    N : int, optional
        Number of peaks (GMM components). Default: ``adata.uns['N_rainbow_peak']``.
    init_params : str, default 'kmeans'
        Initialisation of the GMM: any ``init_params`` option of ``GaussianMixture`` ('kmeans', 'k-means++', 'random',
        ...), or 'custom-uniform', which assumes equally large peaks and starts the means at the
        ``(k + 0.5) / N`` percentiles of every channel, the weights at 1/N and the precisions from the beads in the
        k-th slice of the mean rank across channels.
    random_state : int, numpy.random.RandomState or None, default None
        Seed of ``GaussianMixture``. None gives a different fit in every call.
    n_init : int, default 10
        Number of GMM initialisations; the best one is kept.
    max_iter : int, default 50
        Maximum number of EM iterations per initialisation.
    clipminperc, clipmaxperc : float, default 0.1 and 99.9
        Percentiles (per channel, in percent) to which the arcsinh-transformed data are clipped before the fit, so
        that single extreme events do not get their own component.
    warningthreshold : float, default 1.5
        The fit counts as failed if the smallest peak share is below ``1 / N / warningthreshold`` and the largest
        above ``warningthreshold / N``, or if a component has no beads at all.
    fct_cofactor : callable, default `_GMM_choose_asinh_cofactor`
        Function ``fct_cofactor(adata) -> float`` that chooses the arcsinh cofactor (raw units). The default sets it
        so that the geometric mean over channels of the 90th percentiles maps to arcsinh value 7.
    Nretry : int, default 3
        If the fit counts as failed (see `warningthreshold`), warn and fit again, at most `Nretry` times (an int
        `random_state` is increased by 1 per retry); after the last attempt warn with the peak shares. Set to 0 if the
        peaks of the bead type are not equally large (an empty component still gives the warning).
    cofactor : float, optional
        Arcsinh cofactor (raw units). Default: chosen by `fct_cofactor`.
    order_channels : list of str, optional
        Peak ordering. None (default): peaks are sorted by their mean rank over all fluorescence channels, from dim to
        bright (rainbow beads have the same brightness order in every channel). A list of K channels: peak k is the
        component with the highest mean in ``order_channels[k]``, the remaining N - K peaks follow in the default order
        (e.g. calibration beads with one laser-specific positive peak per channel plus a negative peak, whose rank
        order is not consistent across channels and files). If these channels do not pick K distinct components, the
        default order is used, a `GMMAssignmentWarning` is issued and ``uns['GMM_order_ok']`` is False.

    Returns
    -------
    None
        `adata` is changed in place: ``obs['GMM_label']`` (int, peak 0..N-1 in the order above) and ``obs['GMM_llh']``
        (log-likelihood of the bead under the GMM, on the arcsinh scale), and in ``uns``: 'GMM_means' (peak means
        on the raw scale, ``sinh(mean) * cofactor``, shape (peak, channel)), 'GMM_asinh_means' (peak, channel),
        'GMM_asinh_covariances' (peak, channel, channel), 'GMM_asinh_stds' (peak, channel), 'GMM_asinh_cofactor'
        and 'GMM_order_ok' (bool). Row k of every array belongs to ``GMM_label == k``; channels are
        ``cytoconfig.channels_fluorescence``. A retry overwrites all of them.

    Raises
    ------
    ValueError
        If a channel of `order_channels` is not a fluorescence channel.

    Warns
    -----
    GMMAssignmentWarning
        If the fit counts as failed (see `warningthreshold`, `Nretry`), or `order_channels` do not pick distinct
        peaks.

    See Also
    --------
    get_rainbow_GMM_gate : likelihood gate based on this fit.
    plot_rainbow_channel_GMM_single, plot_GMM_llh : plots to check the fit.
    reorder_peaks : change the peak order afterwards.

    Examples
    --------
    >>> fit_rainbow_GMM(adata, random_state=0)
    >>> adata = adata[get_rainbow_GMM_gate(adata, cutoff=-2, dropmaxpercent=10)].copy()
    >>> fit_rainbow_GMM(adata, random_state=0)
    """
    if N is None:
        N = adata.uns["N_rainbow_peak"]
    
    # prepare data
    asinh_cofactor = fct_cofactor(adata) if cofactor is None else cofactor
    X = np.arcsinh(adata[:,adata.uns["cytoconfig"].channels_fluorescence].layers["raw"] / asinh_cofactor)
    X = np.clip(X, a_min=np.percentile(X, clipminperc, axis=0, keepdims=True),
                   a_max=np.percentile(X, clipmaxperc, axis=0, keepdims=True))

    # fit GMM
    if init_params=="custom-uniform":
        # assumes that all peaks are present in similar proportions
        # initializes means with respective percentiles, component weights with 1/N
        means_init = np.percentile(X, ((np.arange(N)+.5) / N)*100, axis=0)
        weights_init = np.ones((N))/N

        def single_precision(X, N, i):
            order = np.argsort(np.argsort(X, axis=0), axis=0).mean(1)
            low = np.percentile(order, (i)*100/N, axis=0, keepdims=True)
            high = np.percentile(order, (i+1)*100/N, axis=0, keepdims=True)
            mask = (order>=low) & (order<=high)
            precision = np.linalg.inv(np.cov(X[mask].T))
            return precision
        precisions_init = np.asarray([single_precision(X, N, i) for i in range(N)])

        gm = GaussianMixture(n_components=N, random_state=random_state, n_init=n_init, max_iter=max_iter, init_params="kmeans",
                             means_init=means_init, weights_init=weights_init, precisions_init=precisions_init)
    else:
        gm = GaussianMixture(n_components=N, random_state=random_state, n_init=n_init, max_iter=max_iter, init_params=init_params)
    gm.fit(X)
    labels = gm.predict(X)

    # get proper label order correspondence (label_order[component] = new label)
    label_order = np.argsort(np.argsort(np.argsort(np.argsort(gm.means_, 0), 0).mean(-1)))
    order_ok = True
    if order_channels is not None:
        idx = pd.Index(adata.uns["cytoconfig"].channels_fluorescence).get_indexer(order_channels)
        if (idx < 0).any():
            raise ValueError(f"order_channels not found in channels_fluorescence: {np.asarray(order_channels)[idx < 0]}")
        first = list(np.argmax(gm.means_[:, idx], axis=0))
        if len(set(first)) == len(first):
            rest = [i for i in np.argsort(label_order) if i not in first]
            label_order = np.argsort(first + rest)
        else:
            order_ok = False
            warnings.warn(f"order_channels do not pick distinct GMM peaks for sample {adata.uns.get('name_uid')}, "
                          "using the default peak order!", GMMAssignmentWarning)
    labels = label_order[labels]
    # store data
    adata.obs["GMM_label"] = labels
    adata.obs["GMM_llh"] = gm.score_samples(X)
    # also store means as raw
    adata.uns["GMM_means"] = np.sinh(gm.means_[np.argsort(label_order)]) * asinh_cofactor
    adata.uns["GMM_asinh_means"] = gm.means_[np.argsort(label_order)]
    adata.uns["GMM_asinh_covariances"] = gm.covariances_[np.argsort(label_order)]
    adata.uns["GMM_asinh_cofactor"] = asinh_cofactor
    adata.uns["GMM_order_ok"] = order_ok
    adata.uns["GMM_asinh_stds"] = np.sqrt(adata.uns["GMM_asinh_covariances"][:, np.arange(gm.means_.shape[1]), np.arange(gm.means_.shape[1])])
    
    # Throw warning if peak class proportions seem too uneven, or if a component got no beads at all
    # (bincount with minlength, so that empty components count with share 0)
    c = np.bincount(labels, minlength=N)
    u = np.arange(N)
    label_shares = c/c.sum()
    uneven = label_shares.min()<1/N/warningthreshold and label_shares.max()>warningthreshold/N
    if uneven or (c == 0).any():
        if Nretry>0:
            warnstr = f"GMM rainbow bead class assignment may not have worked properly for sample {adata.uns.get('name_uid')}, trying again ({Nretry})!"
            warnings.warn(warnstr, GMMAssignmentWarning)
            # an int seed would repeat the identical fit, so derive a new deterministic seed for every retry;
            # None and np.random.RandomState instances give a different fit anyway
            retry_state = random_state + 1 if isinstance(random_state, (int, np.integer)) else random_state
            fit_rainbow_GMM(adata, N=N, init_params=init_params, random_state=retry_state, n_init=n_init, max_iter=max_iter,
                    clipminperc=clipminperc, clipmaxperc=clipmaxperc, warningthreshold=warningthreshold, fct_cofactor=fct_cofactor, Nretry=Nretry-1, cofactor=cofactor,
                    order_channels=order_channels)
        else:
            warnstr = f"GMM rainbow bead class assignment may not have worked properly for sample {adata.uns.get('name_uid')}!"
            warnstr += f"\nSmallest peak class is {u[np.argmin(c)]} with share {label_shares[np.argmin(c)]:.2f}"
            warnstr += f"\nLargest peak class is {u[np.argmax(c)]} with share {label_shares[np.argmax(c)]:.2f}"
            warnings.warn(warnstr, GMMAssignmentWarning)


def _stats_per_group(x, group, clippercent=None):
    """Mean, median, std of `x` per group (DataFrame, index group), optionally clipped to [p, 100-p] percentiles."""
    def fct_clip(x):
        if clippercent is None:
            return x
        return np.clip(x, a_min=np.percentile(x, clippercent), a_max=np.percentile(x, 100-clippercent))
    us = np.unique(group)
    means = [np.mean(fct_clip(x[group==u])) for u in us]
    medians = [np.median(fct_clip(x[group==u])) for u in us]
    stds = [np.std(fct_clip(x[group==u])) for u in us]
    return pd.DataFrame([means, medians, stds], columns=us, index=["mean","median","std"]).T

def _grouped_percentile(x, group, percentile):
    """Series (index: group) of the `percentile`-th percentile of `x` per group."""
    us = np.unique(group)
    percentiles = [np.percentile(x[group==u], percentile) for u in us]
    return pd.Series(percentiles, index=us)

def _get_rainbow_GMM_cutoff(adata, cutoff=-2., clippercent=5, dropmaxpercent=10):
    """Log-likelihood cutoff per GMM label (Series, index: label); see `get_rainbow_GMM_gate` for the parameters."""
    df_peaks = _stats_per_group(adata.obs["GMM_llh"], adata.obs["GMM_label"], clippercent=clippercent)
    percentiles = _grouped_percentile(adata.obs["GMM_llh"], adata.obs["GMM_label"], dropmaxpercent)
    llh_cutoff = df_peaks["mean"] + cutoff*df_peaks["std"]
    mask = llh_cutoff>percentiles
    llh_cutoff[mask] = percentiles[mask]
    return llh_cutoff

def get_rainbow_GMM_gate(adata, cutoff=-2., clippercent=5, dropmaxpercent=10):
    """Mask of the beads that fit their peak well under the GMM of `fit_rainbow_GMM`.

    For every peak, the mean and standard deviation of ``obs['GMM_llh']`` are computed after clipping it to the
    [`clippercent`, 100 - `clippercent`] percentiles of the peak (less influence of outliers). Beads whose
    log-likelihood is below ``mean + cutoff * std`` of their peak are dropped, but at most `dropmaxpercent` percent of
    every peak: if the cutoff would drop more, it is lowered to the `dropmaxpercent`-th percentile of the peak.

    Parameters
    ----------
    adata : anndata.AnnData
        Rainbow bead measurement after `fit_rainbow_GMM` (uses ``obs['GMM_llh']`` and ``obs['GMM_label']``).
    cutoff : float, default -2.0
        Cutoff in standard deviations of the peak's log-likelihood, relative to its mean (negative: below the mean).
    clippercent : float, default 5
        Percentiles (in percent, at both ends) to which the log-likelihoods are clipped for the mean and standard
        deviation.
    dropmaxpercent : float, default 10
        Maximum share (in percent) of every peak that is dropped.

    Returns
    -------
    pandas.Series
        Boolean mask (index: ``adata.obs_names``), True for the beads to keep. Strict: a bead exactly at the cutoff is
        dropped. `adata` is not changed.

    See Also
    --------
    plot_GMM_llh : histograms of the log-likelihood with these cutoffs.

    Examples
    --------
    >>> adata = adata[get_rainbow_GMM_gate(adata, cutoff=-2, dropmaxpercent=10)].copy()
    """
    llh_cutoff = _get_rainbow_GMM_cutoff(adata, cutoff, clippercent, dropmaxpercent)
    mask = adata.obs["GMM_llh"] > llh_cutoff.loc[adata.obs["GMM_label"]].to_numpy()
    return mask
