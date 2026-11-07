"""Plots to check the rainbow bead GMM fit, the likelihood gate and the result of a `RainbowCalibrator` fit."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import palette_20, get_cmap, cmap_to_legendhandles
from .gmm import _get_rainbow_GMM_cutoff, _stats_per_group


def _get_max_separation_channel(adata):
    """Index (in ``channels_fluorescence``) of the channel where neighbouring GMM peaks are separated best.

    For every channel, the distance between consecutive peaks (in label order, arcsinh scale) divided by their combined
    standard deviation (clipped to its 20th and 80th percentile over all channels); the channel with the largest minimum
    over peak pairs is returned.
    """
    diff = np.abs(adata.uns["GMM_asinh_means"][:-1] - adata.uns["GMM_asinh_means"][1:])
    std = np.sqrt(adata.uns["GMM_asinh_stds"][:-1]**2 + adata.uns["GMM_asinh_stds"][1:]**2)
    std = np.clip(std, a_min=np.percentile(std.flatten(), 20), a_max=np.percentile(std.flatten(), 80))
    z = diff/std
    return np.argmax(z.min(axis=0))

def plot_rainbow_channel_GMM_single(adata, i=None, ax=None, bins=None, savepath=""):
    """Plot the histogram of one fluorescence channel of a rainbow bead measurement, coloured by GMM peak.

    Parameters
    ----------
    adata : anndata.AnnData
        Rainbow bead measurement after `fit_rainbow_GMM` (uses ``layers['raw']``, ``obs['GMM_label']``,
        ``uns['GMM_asinh_cofactor']``, ``uns['GMM_asinh_means']``, ``uns['GMM_asinh_stds']``, ``uns['name_uid']``).
    i : int, optional
        Index of the channel in ``adata.uns['cytoconfig'].channels_fluorescence``. Default: the channel in which the
        peaks are separated best.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.
    bins : numpy.ndarray, optional
        Bin edges on the arcsinh scale. Default: 200 bins over the data range.
    savepath : str, default ""
        If not empty, save the current figure there (dpi 200) and close it.

    Returns
    -------
    None

    Notes
    -----
    The x axis is ``arcsinh(raw / uns['GMM_asinh_cofactor'])``, labelled with the channel name; the title is
    ``uns['name_uid']``.
    """
    if i is None:
        i = _get_max_separation_channel(adata)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4), layout="tight")
    X = np.arcsinh(adata[:,adata.uns["cytoconfig"].channels_fluorescence[i]].layers["raw"][:,0] / adata.uns["GMM_asinh_cofactor"])
    if bins is None:
        bins = np.linspace(X.min()-.2, X.max()+.2, 200)
    sns.histplot(x=X, hue=adata.obs["GMM_label"].astype(str), hue_order=sorted(adata.obs["GMM_label"].unique().astype(str)),
                 bins=bins, fill=False, element="step", ax=ax, palette=palette_20[:len(adata.obs["GMM_label"].unique())])
    ax.set_xlabel(adata.uns["cytoconfig"].channels_fluorescence[i], size=20)
    ax.set_xlim([bins.min(), bins.max()])
    ax.set_title(adata.uns["name_uid"], size=20)

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.close()


def plot_GMM_llh(adata, cutoff=-2., clippercent=5, dropmaxpercent=10, ax=None, plotminstd=4, plotmaxstd=3):
    """Plot the GMM log-likelihood histogram per peak, with the cutoffs of `get_rainbow_GMM_gate`.

    Parameters
    ----------
    adata : anndata.AnnData
        Rainbow bead measurement after `fit_rainbow_GMM` (uses ``obs['GMM_llh']`` and ``obs['GMM_label']``).
    cutoff, clippercent, dropmaxpercent : float, default -2.0, 5 and 10
        Gate settings, as in `get_rainbow_GMM_gate`; use the values of the gate you apply.
    ax : matplotlib.axes.Axes, optional
        Axes for the histograms. Default: a new figure.
    plotminstd : float, default 4
        Lower plot limit in standard deviations below the peak mean (at least ``-cutoff + 0.5``), minimum over peaks.
    plotmaxstd : float, default 3
        Upper plot limit in standard deviations above the peak mean, maximum over peaks.

    Returns
    -------
    None

    Notes
    -----
    The log-likelihoods are clipped to the plot range, so the outermost bins also count the beads beyond it. The
    cutoff lines (dashed, one per peak) are drawn with ``plt.axvline``, i.e. on the current axes, which is not
    necessarily `ax`: call ``plt.sca(ax)`` first when passing `ax`.
    """
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
    df_peaks = _stats_per_group(adata.obs["GMM_llh"], adata.obs["GMM_label"], clippercent=clippercent)
    
    plotmin = (df_peaks["mean"]-df_peaks["std"]*max(plotminstd, -cutoff+.5)).min()
    plotmax = (df_peaks["mean"]+df_peaks["std"]*plotmaxstd).max()
    
    sns.histplot(x=np.clip(adata.obs["GMM_llh"], plotmin, plotmax), hue=adata.obs["GMM_label"].astype(str),
                 hue_order=sorted(adata.obs["GMM_label"].unique().astype(str)),
                 bins=np.linspace(plotmin, plotmax, 100), fill=False, element="step", ax=ax,
                 palette=palette_20[:df_peaks.shape[0]])
    ax.set_xlim([plotmin, plotmax])
    
    llh_cutoff = _get_rainbow_GMM_cutoff(adata, cutoff, clippercent, dropmaxpercent)
    for i, v in enumerate(llh_cutoff):
        plt.axvline(v, color=palette_20[i], linestyle="--")


def plot_eval_RBInt_fit(rbint, adatas, channels=None, legendkey=None, bins=300, ylim=None):
    """Plot rainbow bead histograms before and after calibration, to check a `RainbowCalibrator` fit.

    One row per channel, three columns: raw data, calibrated without the fitted shift, and calibrated with the shift.
    After a good fit, the peaks of all measurements of a bead type coincide in the calibrated columns.

    Parameters
    ----------
    rbint : RainbowCalibrator
        Fitted calibrator.
    adatas : list of anndata.AnnData
        Rainbow bead measurements known to `rbint` (their batch, uid and rainbow type must be in the model).
        Changed in place: layers 'calibrated_rb' (with shift) and 'calibrated_rb_noshift' are added (overwritten),
        see `RainbowCalibrator.add_calibrated_rainbow`.
    channels : list of str, optional
        Channels to plot. Default: all fluorescence channels of the model.
    legendkey : str, optional
        Key of ``adata.uns`` by which the histograms are coloured, with a legend in the first column (e.g.
        'rainbow_batch'). Default: matplotlib's colour cycle, no legend.
    bins : int, default 300
        Number of histogram bins (per measurement).
    ylim : tuple of float, optional
        y limits (density) for all axes.

    Returns
    -------
    None

    Notes
    -----
    All columns show ``arcsinh(x / cofactor)`` with the model's cofactor of the measurement's rainbow type
    (`RainbowCalibrator.apply_ArcSinh`); histograms are normalised to density.
    """
    if channels is None:
        channels = rbint.rfim.channels
    
    for ad in adatas:
        rbint.add_calibrated_rainbow(ad, include_shift=True, addlayer="calibrated_rb")
        rbint.add_calibrated_rainbow(ad, include_shift=False, addlayer="calibrated_rb_noshift")
    
    fig, ax = plt.subplots(len(channels), 3, figsize=(20, len(channels)*3.5), layout="tight")
    if len(ax.shape)<2: ax = ax[None]
    
    get_data = lambda ad, l, c: rbint.apply_ArcSinh(ad[:,c].layers[l][:,0], ad.uns["rainbow_type"])
    
    cmap = get_cmap([ad.uns[legendkey] for ad in adatas]) if legendkey is not None else None

    ax[0,0].set_title("Raw", size=25)
    ax[0,1].set_title("Calibrated w/o Shift", size=25)
    ax[0,2].set_title("Calibrated", size=25)
    
    for i, channel in enumerate(channels):
        for ad in adatas:
            color = None if legendkey is None else cmap[ad.uns[legendkey]]
            ax[i, 0].hist(get_data(ad, "raw", channel), bins=bins, fill=False, density=True, histtype="step", color=color)
            ax[i, 1].hist(get_data(ad, "calibrated_rb_noshift", channel), bins=bins, fill=False, density=True, histtype="step", color=color)
            ax[i, 2].hist(get_data(ad, "calibrated_rb", channel), bins=bins, fill=False, density=True, histtype="step", color=color)
        ax[i, 0].set_ylabel(channel, size=25)
        if cmap is not None:
            ax[i, 0].legend(handles=cmap_to_legendhandles(cmap))
        if ylim is not None:
            for ax_ in ax[i]:
                ax_.set_ylim(ylim)


