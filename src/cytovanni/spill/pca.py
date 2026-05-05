from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SingleSpectrumPCAFitter():
    """ Fit PCA to single dye spectra.
    """
    def __init__(self, spectra, N=5, name=""):
        """ 
            
            :param spectra: pd.DataFrame. Fitted spectra for the same dye from different measurements, columns are the channels, index the measurements.
            
            :param N: int. Max number of principal components that should be fitted.
            
            :param name: str. Optional name.
        """
        spectra = spectra[np.isnan(spectra).sum(1)==0]
        self.channels = np.asarray(spectra.columns)
        self.obs = np.asarray(spectra.index)
        self.spectra = np.asarray(spectra)
        self.spectra_means = self.spectra.mean(0, keepdims=True)
        self.spectra_zeromean = self.spectra - self.spectra_means
        self.N = min(N, self.spectra.shape[0])
        self.name = name
        
        self.is_dummy = self.spectra.shape[0]<=1
        
        self.fit()
    
    def fit(self):
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
        """ Plot all entries against those smoothed using N components.
        """
        for i in range(self.N):
            plt.scatter(self.spectra_zeromean.flatten(), self.spectra_smoothed[i].flatten(), label=f"N={i+1}")
        plt.legend()
    
    def plot_expvar(self):
        """ Plot explained variance share per component.
        """
        plt.scatter(np.arange(self.N)+1, self.pca.explained_variance_ratio_, label=self.name)
        plt.plot(np.arange(self.N)+1, self.pca.explained_variance_ratio_)
        plt.xticks(np.arange(self.N)+1)
        plt.axhline(.02, color="gray")
        plt.xlabel("N")
        plt.ylabel("Explained Variance")
        plt.ylim([0,None])
    
    def get_smoothed(self, i):
        """ Get spectra, smoothed by only using fixed number of principal component contributions.
        """
        return pd.DataFrame((self.spectra_smoothed[i-1] if i>0 else np.zeros((len(self.obs), len(self.channels)))) + self.spectra_means,
                            columns=self.channels, index=self.obs)
    
    def get_real(self):
        """ Get real spectra.
        """
        return pd.DataFrame(self.spectra, columns=self.channels, index=self.obs)

    def add_explained_share_above_cutoff(self, cutoff=.01):
        """ Add explained share above cutoff (SAC).
            Score is share of spectrum measurements that are off by more than cutoff,
            summed across channels.
            Explained SAC referrs to the improvement one gains by including a component,
            vs. only including components up to it.
        """
        def get_single(i):
            val = np.abs(self.get_real() - self.get_smoothed(i))>cutoff
            return val.mean().sum()
        self.total_SAC = get_single(0)
        expl = np.asarray([get_single(n) for n in range(self.N+1)])
        self.explained_SAC = expl[:-1] - expl[1:]
