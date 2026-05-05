import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import get_cmap, label_axis_ArcSinh, unrolled_subplots, cmap_to_legendhandles
from ..utils import base_plot_NxN_ds, base_plot_NxN


def plot_eval_histogram(adatas, unmxkey, uid, key_uid, key_color=None, key_subset=None, subset=[], xlim=[-1e4, 3e5], ylim=[1e-3,None], Nbin=100, savepath="", log=True, cofactor=1500):
    """ Plot histograms for every marker to evaluate integration.

        :param adatas: iterable. List of all sample adatas.

        :param unmxkey: str. Key for the unmixing in adata.obsm that should be used.

        :param uid: str. Subset adatas to only those where adata.uns[key_uid] is equal to this, to plot only aliquots of the same sample.

        :param key_uid: str. Key for adata.uns from which to get unique id.

        :param key_color: None, str. Color based on this key in adata.uns.

        :param key_subset: None, str. If given, also subset adatas based on whether adata.uns[key_subset] is in subset.

        :param subset: list. Subset to use with key_subset.

        :param xlim: iterable. x axis limits in fluorescence units.

        :param ylim: iterable. y axis limits.

        :param Nbin: int. Number of bins for the histogram.

        :param savepath: str. If given, saves and closes the plot.

        :param log: bool. Whether to use log of y axis.

        :param cofactor: float. Cofactor to use for arcsinh transformation.
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
    """ Plot histograms for every marker to evaluate integration.

        :param adatas: iterable. List of all sample adatas.

        :param unmxkey: str. Key for the unmixing in adata.obsm that should be used.

        :param uid: str. Subset adatas to only those where adata.uns[key_uid] is equal to this, to plot only aliquots of the same sample.

        :param key_uid: str. Key for adata.uns from which to get unique id.

        :param key_color: None, str. Color based on this key in adata.uns.

        :param key_subset: None, str. If given, also subset adatas based on whether adata.uns[key_subset] is in subset.

        :param subset: list. Subset to use with key_subset.

        :param axlim: iterable. Axis limits in fluorescence units.

        :param savepath: str. If given, saves and closes the plot.

        :param cofactor: float. Cofactor to use for arcsinh transformation.

        :param datashader: bool. Whether to use the datashader implementation of NxN plots.
    """
    adatas_use = [ad for ad in adatas if ad.uns[key_uid]==uid]
    if key_subset is not None:
        adatas_use = [ad for ad in adatas_use if ad.uns[key_subset] in subset]
    xs = [np.arcsinh(ad.obsm[unmxkey] / cofactor) for ad in adatas_use]
    labels = np.arange(len(xs)) if key_color is None else np.asarray([ad.uns[key_color] for ad in adatas_use])
    colors = list(get_cmap(labels).values())
    axlim_ash = (np.arcsinh(axlim[0]/cofactor), np.arcsinh(axlim[1]/cofactor))

    if datashader:
        base_plot_NxN_ds(xs, colors=colors, axlim=axlim_ash, savepath=savepath)
    else:
        base_plot_NxN(xs, colors, labels, savepath=savepath, suptitle=uid, axlim=axlim_ash)
