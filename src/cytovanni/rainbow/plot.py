import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import palette_20, get_cmap, cmap_to_legendhandles
from .gmm import _get_rainbow_GMM_cutoff, _stats_per_group


def _get_max_separation_channel(adata):
    """ Get channel where separation between peaks is clearest.
        Calculates the z-score of the difference between each peak and the next highest,
        min over differences, argmax over channels.
    """
    diff = np.abs(adata.uns["GMM_asinh_means"][:-1] - adata.uns["GMM_asinh_means"][1:])
    std = np.sqrt(adata.uns["GMM_asinh_stds"][:-1]**2 + adata.uns["GMM_asinh_stds"][1:]**2)
    std = np.clip(std, a_min=np.percentile(std.flatten(), 20), a_max=np.percentile(std.flatten(), 80))
    z = diff/std
    return np.argmax(z.min(axis=0))

def plot_rainbow_channel_GMM_single(adata, i=None, ax=None, bins=None, savepath=""):
    """ Plot histogram along single channel i, if not set default to the one with cleanest separation of peaks.
        Split by assignment into rainbow bead peaks.
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
    """ Plot llh histograms per GMM label, together with GMM gating cutoff.
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


