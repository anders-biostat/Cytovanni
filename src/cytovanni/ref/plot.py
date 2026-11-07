"""Plots to check a standardisation: all measurements (aliquots) of one sample overlaid, per marker or NxN."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import get_cmap, label_axis_ArcSinh, unrolled_subplots, cmap_to_legendhandles
from ..utils import base_plot_NxN_ds, base_plot_NxN


def plot_eval_histogram(adatas, unmxkey, uid, key_uid, key_color=None, key_subset=None, subset=[], xlim=[-1e4, 3e5], ylim=[1e-3,None], Nbin=100, savepath="", log=True, cofactor=1500):
    """Plot one histogram per marker, overlaying all measurements of one sample, to evaluate a standardisation.

    After a good standardisation the aliquots of the same sample measured in different batches should give
    overlapping histograms. One panel per marker (column of ``adata.obsm[unmxkey]``) on the arcsinh scale with
    axis labels in linear units, plus a first panel with the colour legend; the sample id is the title.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        Measurements to choose from; each needs ``obsm[unmxkey]`` and ``uns[key_uid]`` (and the other uns keys used).
    unmxkey : str
        Key of the unmixed data in ``adata.obsm``: a pandas.DataFrame (events, stain) on a linear scale (e.g. the
        standardised unmixed data); it is arcsinh-transformed here. All selected AnnData objects need the same
        columns in the same order (the first one's column names label the panels).
    uid : str
        Sample to plot: only AnnData objects with ``adata.uns[key_uid] == uid`` are used.
    key_uid : str
        Key in ``adata.uns`` holding the sample id.
    key_color : str, optional
        Key in ``adata.uns`` to colour by (e.g. the batch); measurements with the same value share a colour. Default:
        one colour per measurement.
    key_subset : str, optional
        If given, additionally keep only AnnData objects with ``adata.uns[key_subset]`` in `subset`.
    subset : list, default []
        Allowed values for `key_subset`.
    xlim : list of float, default [-1e4, 3e5]
        Range of the histogram bins, in linear units (transformed with the same cofactor).
    ylim : list of float, default [1e-3, None]
        y-axis limits (density).
    Nbin : int, default 100
        Number of bin edges between the `xlim` bounds.
    savepath : str, default ""
        If given, save the figure there (dpi 200) and close it. Otherwise the figure stays open.
    log : bool, default True
        Logarithmic y axis.
    cofactor : float, default 1500
        Cofactor of the arcsinh transform, in the units of the data.

    Raises
    ------
    IndexError
        If no AnnData object matches `uid` (and `subset`).
    """
    adatas_use = [ad for ad in adatas if ad.uns[key_uid]==uid]
    if key_subset is not None:
        adatas_use = [ad for ad in adatas_use if ad.uns[key_subset] in subset]
    xs = [np.arcsinh(ad.obsm[unmxkey] / cofactor) for ad in adatas_use]
    labels = np.arange(len(xs)) if key_color is None else np.asarray([ad.uns[key_color] for ad in adatas_use])
    cmap = get_cmap(labels)
    bins = np.linspace(np.arcsinh(xlim[0]/cofactor), np.arcsinh(xlim[1]/cofactor), Nbin)

    fig, ax = unrolled_subplots(xs[0].shape[1]+1, Ncol=3, elsize=(7,4))
    ax[0].axis('off')
    ax[0].legend(handles=cmap_to_legendhandles(cmap), loc="upper left", ncols=int(np.ceil(len(cmap)/10)))
    for i in range(len(ax)-1):
        for j, x in enumerate(xs):
            ax[i+1].hist(x.iloc[:,i], bins=bins, fill=False, histtype="step", density=True, color=cmap[labels[j]])
        if log: ax[i+1].set_yscale("log")
        ax[i+1].set_ylim(ylim)
        ax[i+1].set_xlabel(xs[0].columns[i], size=20)
        label_axis_ArcSinh(ax[i+1], cofactor, minpower=3)
    fig.suptitle(uid, size=25)
    fig.set_layout_engine("constrained")
    if savepath:
        plt.savefig(savepath, dpi=200)
        plt.close()


def plot_eval_NxN(adatas, unmxkey, uid, key_uid, key_color=None, key_subset=None, subset=[], axlim=[-1e4, 3e5], savepath="", cofactor=1500, datashader=True):
    """Plot all marker pairs as NxN scatter plots, overlaying all measurements of one sample.

    Same selection and colouring as `plot_eval_histogram`, but every pair of markers is shown as a 2-D scatter plot
    on the arcsinh scale (`cytovanni.utils.base_plot_NxN_ds` or `cytovanni.utils.base_plot_NxN`).

    Parameters
    ----------
    adatas : list of anndata.AnnData
        Measurements to choose from; each needs ``obsm[unmxkey]`` and ``uns[key_uid]`` (and the other uns keys used).
    unmxkey : str
        Key of the unmixed data in ``adata.obsm``: a pandas.DataFrame (events, stain) on a linear scale; it is
        arcsinh-transformed here.
    uid : str
        Sample to plot: only AnnData objects with ``adata.uns[key_uid] == uid`` are used.
    key_uid : str
        Key in ``adata.uns`` holding the sample id.
    key_color : str, optional
        Key in ``adata.uns`` to colour by; measurements with the same value share a colour. Default: one colour per
        measurement.
    key_subset : str, optional
        If given, additionally keep only AnnData objects with ``adata.uns[key_subset]`` in `subset`.
    subset : list, default []
        Allowed values for `key_subset`.
    axlim : list of float, default [-1e4, 3e5]
        Limits of both axes, in linear units (transformed with the same cofactor).
    savepath : str, default ""
        If given, save the plot there. Without it, the matplotlib version leaves the figure open and the datashader
        version returns the image.
    cofactor : float, default 1500
        Cofactor of the arcsinh transform, in the units of the data.
    datashader : bool, default True
        Use the rasterised datashader version (fast for many events, no axis ticks, no title or legend; needs the
        ``plot`` extra) instead of matplotlib.

    Returns
    -------
    PIL.Image.Image or None
        The datashader image if `datashader` is True and no `savepath` is given (shown inline in Jupyter), else None.
    """
    adatas_use = [ad for ad in adatas if ad.uns[key_uid]==uid]
    if key_subset is not None:
        adatas_use = [ad for ad in adatas_use if ad.uns[key_subset] in subset]
    xs = [np.arcsinh(ad.obsm[unmxkey] / cofactor) for ad in adatas_use]
    labels = np.arange(len(xs)) if key_color is None else np.asarray([ad.uns[key_color] for ad in adatas_use])
    cmap = get_cmap(labels)
    colors = [cmap[l] for l in labels] # one colour per dataset, datasets with the same label share it
    axlim_ash = (np.arcsinh(axlim[0]/cofactor), np.arcsinh(axlim[1]/cofactor))

    if datashader:
        return base_plot_NxN_ds(xs, colors=colors, axlim=axlim_ash, savepath=savepath)
    else:
        base_plot_NxN(xs, colors, labels, savepath=savepath, suptitle=uid, axlim=axlim_ash)
