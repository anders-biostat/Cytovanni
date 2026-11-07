import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from ..exceptions import GMMAssignmentWarning

def _GMM_choose_asinh_cofactor(adata, percentile=90, target=7, default=1500):
    """ Simple heuristic to choose an appropriate asinh cofactor for the GMM model.
        Takes the geometric mean of the percentile across channels,
        sets cofactor such that it has value target after asinh.
    """
    percentiles = np.percentile(adata[:,adata.uns["cytoconfig"].channels_fluorescence].layers["raw"], percentile, axis=0)
    if (percentiles>0).sum()>0:
        geomm = np.exp(np.log(percentiles[percentiles>0]).mean())
        return geomm/np.sinh(target)
    return default

def fit_rainbow_GMM(adata, N=None, init_params="kmeans",
                    random_state=None, n_init=10, max_iter=50,
                    clipminperc=.1, clipmaxperc=99.9, warningthreshold=1.5, fct_cofactor=_GMM_choose_asinh_cofactor, Nretry=3, cofactor=None, order_channels=None):
    """ Fit simple GMM on arcsinh transformed raw intensities, assign peak to every bead.
        Adds class label 'GMM_label' and log-likelihood under the GMM model 'GMM_llh' to adata.obs.
        Adds means and covariances of the GMM model to adata.uns, in the same peak order as GMM_label
        (array index k belongs to GMM_label k), and 'GMM_order_ok' (see order_channels).
        
        :param adata: AnnData. Data to be fitted.
        
        :param N: int. Number of rainbow peaks in the sample. Optional if 'N_rainbow_peak' is set in adata.uns
        
        :param init_params: str. Either 'custom-uniform' or available options of GaussianMixture ('kmeans' etc.), custom uniform assumes equal peak sizes for the rainbow beads and initializes with percentiles along each channel.
        
        :param warningthreshold: float. The fit counts as failed if the smallest peak class share is below
            1/N/warningthreshold and the largest above warningthreshold/N, or if any GMM component has no beads at all.

        :param Nretry: int. If the fit counts as failed (see warningthreshold), try again at most Nretry times with a
            new seed, then raise a GMMAssignmentWarning. Disable (0) if rainbow bead peak proportions are not uniform
            across peaks; an empty component still gives the warning.
        
        :param fct_cofactor: function. Function that chooses the arcsinh cofactor, defaults to _GMM_choose_asinh_cofactor.

        :param cofactor: None, float. If given, uses this instead of generating it through fct_cofactor.

        :param order_channels: None or list of channel names. Peak ordering scheme.
            None (default): peaks are sorted by their mean rank across all fluorescence channels, from dim to bright
            (suits rainbow beads, whose peaks have the same brightness order in every channel).
            List of K channels: peak k is the peak with the highest GMM mean in order_channels[k], the remaining N-K peaks
            follow in the default order (e.g. calibration beads with one laser-specific positive peak per channel plus a
            negative peak, whose mean rank order is not consistent across channels and files).
            If these channels do not pick K distinct peaks, the default order is used, a GMMAssignmentWarning is raised
            and adata.uns['GMM_order_ok'] is set to False (True otherwise).
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
    """ Mean, std split by group label.
        Optionally clip values to percentage bounds to not be biased by outliers.
    """
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
    """ Get percentile of x by group label.
    """
    us = np.unique(group)
    percentiles = [np.percentile(x[group==u], percentile) for u in us]
    return pd.Series(percentiles, index=us)

def _get_rainbow_GMM_cutoff(adata, cutoff=-2., clippercent=5, dropmaxpercent=10):
    """ Get cutoff for rainbow GMM gating.
        
        :param cutoff: float. Only keep beads whose llh under the GMM model is above mean + cutoff*std for its class.
        
        :param clippercent: float. Clip values when getting mean/std to clippercent, 100-clippercent percentiles to ignore outliers.
        
        :param dropmaxpercent: float. Drop at most dropmaxpercent of each class and adjust llh cutoffs accordingly.
    """
    df_peaks = _stats_per_group(adata.obs["GMM_llh"], adata.obs["GMM_label"], clippercent=clippercent)
    percentiles = _grouped_percentile(adata.obs["GMM_llh"], adata.obs["GMM_label"], dropmaxpercent)
    llh_cutoff = df_peaks["mean"] + cutoff*df_peaks["std"]
    mask = llh_cutoff>percentiles
    llh_cutoff[mask] = percentiles[mask]
    return llh_cutoff

def get_rainbow_GMM_gate(adata, cutoff=-2., clippercent=5, dropmaxpercent=10):
    """ Get mask for rainbow GMM gating.
        
        For each rainbow bead peak, calculate mean and standard deviation of the log-likelihood.
        The highest and lowest clippercent percent of llh get clipped to reduce the influence of outliers.
        Drop all beads whose llh is cutoff standard deviations below the mean of its peak.
        But drop at most dropmaxpercent percent of every peak.
        
        :param cutoff: float. Only keep beads whose llh under the GMM model is above mean + cutoff*std for its class.
        
        :param clippercent: float. Clip values when getting mean/std to clippercent, 100-clippercent percentiles to ignore outliers. 5 seems reasonable tradeoff.
        
        :param dropmaxpercent: float. Drop at most dropmaxpercent of each class and adjust llh cutoffs accordingly.
    """
    llh_cutoff = _get_rainbow_GMM_cutoff(adata, cutoff, clippercent, dropmaxpercent)
    mask = adata.obs["GMM_llh"] > llh_cutoff.loc[adata.obs["GMM_label"]].to_numpy()
    return mask
