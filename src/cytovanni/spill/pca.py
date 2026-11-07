"""PCA of the spectra of one dye measured in several single-stain measurements."""
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SingleSpectrumPCAFitter():
    """PCA of several measured spectra of one dye.

    Drops spectra that contain NaN, centres the remaining spectra on their mean and fits a PCA (scikit-learn) with up
    to `N` components, so that ``spectra ~ mean + embedding @ components``. The fit runs on construction. With a single
    spectrum, or spectra that are all identical, the fit is a dummy with no components (``is_dummy``, ``N = 0``).

    Parameters
    ----------
    spectra : pandas.DataFrame
        Spectra of the same dye from different measurements, shape (measurement, channel): index the measurements,
        columns the channels (e.g. normalised spectra from `SingleStainSpilloverCollection`).
    N : int, default 5
        Maximum number of principal components. Capped at the number of spectra minus one and at the number of
        channels: n spectra centred on their mean span at most n-1 dimensions, a further component would have zero
        variance (and a NaN or huge normed embedding). Also capped at the numerical rank of the centred spectra
        (singular values with s**2 <= 1e-10 * s_max**2 count as zero): normalised spectra (one channel fixed to 1) and
        other linear dependencies lower the rank.
    name : str, default ""
        Label, used in the legend of `plot_expvar`.

    Attributes
    ----------
    channels : numpy.ndarray
        Channel names (columns of `spectra`).
    obs : numpy.ndarray
        Measurement names of the spectra used (rows without NaN).
    spectra : numpy.ndarray
        The spectra used, shape (measurement, channel).
    spectra_means : numpy.ndarray
        Mean spectrum, shape (1, channel); the offset of the PCA.
    N : int
        Number of fitted components.
    is_dummy : bool
        True if no component could be fitted.
    embedding : numpy.ndarray
        PCA embedding of every spectrum, shape (measurement, component).
    components : numpy.ndarray
        Principal components, shape (component, channel), each with L2 norm 1.
    explained_variance : numpy.ndarray
        Variance per component (ddof=1, as scikit-learn), shape (component,).
    explained_variance_ratio : numpy.ndarray
        Share of the total variance per component, shape (component,).
    total_variance : float
        Total variance of the spectra, summed over channels.
    explained_variance_bychannel : numpy.ndarray
        Variance of each component's contribution per channel, shape (component, channel).
    explained_variance_bychannel_ratio : numpy.ndarray
        `explained_variance_bychannel` divided by the variance of the spectra in that channel (NaN for constant
        channels).
    variance_bychannel : numpy.ndarray
        Variance of the spectra per channel, shape (channel,); not set for a dummy fit.
    max_channel_variance : float
        Largest entry of `variance_bychannel`.
    component_max_effect : numpy.ndarray
        Largest absolute contribution of each component to any channel of any spectrum, shape (component,).
    """
    def __init__(self, spectra, N=5, name=""):
        spectra = spectra[np.isnan(spectra).sum(1)==0]
        self.channels = np.asarray(spectra.columns)
        self.obs = np.asarray(spectra.index)
        self.spectra = np.asarray(spectra)
        self.spectra_means = self.spectra.mean(0, keepdims=True)
        self.spectra_zeromean = self.spectra - self.spectra_means
        # mean-centred spectra have rank <= n_spectra-1; PCA also needs N <= n_channels; null directions are dropped
        self.N = min(N, self.spectra.shape[0]-1, self.spectra.shape[1], self.get_numerical_rank(self.spectra_zeromean))
        self.name = name
        
        self.is_dummy = self.spectra.shape[0]<=1 or self.N<=0
        
        self.fit()
    
    @staticmethod
    def get_numerical_rank(x, rtol=1e-10):
        """Number of singular values s of `x` with s**2 > rtol * max(s)**2.

        Parameters
        ----------
        x : numpy.ndarray
            2-D array.
        rtol : float, default 1e-10
            Relative threshold on the squared singular values.

        Returns
        -------
        int
            Numerical rank; 0 for an empty or all-zero `x`.
        """
        if x.size == 0:
            return 0
        s = np.linalg.svd(x, compute_uv=False)
        if s[0] <= 0:
            return 0
        return int((s**2 > rtol * s[0]**2).sum())
    
    def fit(self):
        """Fit the PCA and set the attributes listed in the class docstring (dummy values if ``is_dummy``).

        Called on construction.
        """
        if self.is_dummy:
            self.N = 0
            self.embedding = np.zeros((len(self.obs),0))
            self.components = np.zeros((0,len(self.channels)))
            self.explained_variance_ratio = np.zeros((0,))
            self.explained_variance = np.zeros((0,))
            self.explained_variance_bychannel = np.zeros((0,len(self.channels)))
            self.total_variance = 0.
            self.max_channel_variance = 0.
            self.explained_variance_bychannel_max = np.zeros((0,))
            self.component_max_effect = np.zeros((0,))
            self.explained_variance_bychannel_ratio = np.zeros((0,len(self.channels)))
        else:
            self.pca = PCA(self.N)
            self.embedding = self.pca.fit_transform(self.spectra_zeromean)
            self.components = self.pca.components_
            self.spectra_smoothed = [self.embedding[:,:i+1] @ self.components[:i+1] for i in range(self.N)]
            self.component_max_effect = np.abs(self.embedding[...,None] * self.components[None,...]).max(0).max(-1)

            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            self.explained_variance = self.pca.explained_variance_
            self.total_variance = self.explained_variance[0] / self.explained_variance_ratio[0]

            self.explained_variance_bychannel = np.var(self.embedding[...,None] * self.components[None], axis=0, ddof=1)
            self.explained_variance_bychannel_max = self.explained_variance_bychannel.max(axis=1)
            self.variance_bychannel = np.var(self.spectra_zeromean, axis=0, ddof=1)
            self.max_channel_variance = self.variance_bychannel.max()
            with np.errstate(invalid='ignore', divide = 'ignore'):
                self.explained_variance_bychannel_ratio = self.explained_variance_bychannel / self.variance_bychannel[None]
    
    def plot_fullscatter(self):
        """Plot every centred spectrum entry against its reconstruction from the first 1, 2, ..., `N` components.

        The plot is empty for a dummy fit.
        """
        for i in range(self.N):
            plt.scatter(self.spectra_zeromean.flatten(), self.spectra_smoothed[i].flatten(), label=f"N={i+1}")
        plt.legend()
    
    def plot_expvar(self):
        """Plot the share of explained variance per component, with a gray reference line at 2%.

        Not available for a dummy fit.
        """
        plt.scatter(np.arange(self.N)+1, self.pca.explained_variance_ratio_, label=self.name)
        plt.plot(np.arange(self.N)+1, self.pca.explained_variance_ratio_)
        plt.xticks(np.arange(self.N)+1)
        plt.axhline(.02, color="gray")
        plt.xlabel("N")
        plt.ylabel("Explained Variance")
        plt.ylim([0,None])
    
    def get_smoothed(self, i):
        """Get the spectra reconstructed from the mean and the first `i` principal components.

        Parameters
        ----------
        i : int
            Number of components to use, 0 to `N`; 0 returns the mean spectrum for every measurement.

        Returns
        -------
        pandas.DataFrame
            Reconstructed spectra, shape (measurement, channel).
        """
        return pd.DataFrame((self.spectra_smoothed[i-1] if i>0 else np.zeros((len(self.obs), len(self.channels)))) + self.spectra_means,
                            columns=self.channels, index=self.obs)
    
    def get_real(self):
        """Get the measured spectra used in the fit as pandas.DataFrame of shape (measurement, channel)."""
        return pd.DataFrame(self.spectra, columns=self.channels, index=self.obs)

    def add_explained_share_above_cutoff(self, cutoff=.01):
        """Compute the share above cutoff (SAC) of the reconstructions and how much each component reduces it.

        The SAC of a reconstruction is the share of measurements whose reconstructed entry differs from the measured one
        by more than `cutoff`, summed over channels. Sets ``total_SAC`` (the SAC of the mean spectrum alone, i.e. 0
        components) and ``explained_SAC`` (numpy.ndarray of shape (component,): the reduction of the SAC when component
        i is added to the components before it).

        Parameters
        ----------
        cutoff : float, default 0.01
            Absolute threshold on the reconstruction error, in units of the spectra (e.g. spillover relative to the main
            channel for normalised spectra).
        """
        def get_single(i):
            val = np.abs(self.get_real() - self.get_smoothed(i))>cutoff
            return val.mean().sum()
        self.total_SAC = get_single(0)
        expl = np.asarray([get_single(n) for n in range(self.N+1)])
        self.explained_SAC = expl[:-1] - expl[1:]
