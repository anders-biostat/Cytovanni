import numpy as np
import matplotlib.pyplot as plt

class EventRateEstimator():
    """ Estimates overall event rate from event times.
        
        Events come from a Poisson point process (with a maximal rate due to doublet generation etc.),
        so the time differences follow an exponential distribution with:
        $f(\Delta t;\kappa) = \kappa \exp(-\kappa \Delta t)$

        Maximizing the combined log-likelihood given samples $\Delta t_i$ yields:
        $\kappa = N / \left(\sum_i^N \ln(\Delta t_i)\right)$

        To avoid undue effect of outliers, we use mean $1/\kappa$ and median $\ln(2)/\kappa$ to get
        the robust estimator:
        $\kappa = \ln(2) / \text{median}(\Delta t_i)$.
    """
    def __init__(self, times):
        self.time = np.sort(np.asarray(times))
        self.timediff = self.time[1:] - self.time[:-1]
        self.N = self.time.shape[0]
        self.kappa = np.log(2) / np.median(self.timediff)
        self.rate = self.kappa
    
    def plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(20,5))
        
        hist, bins, _ = ax[0].hist(self.time, bins=200, fill=False, histtype="step", density=False)
        ax[0].plot((bins[1:] + bins[:-1])/2, self.kappa * (bins[1:] - bins[:-1]), color="black")
        ax[0].set_xlabel("Time", size=20)
        ax[0].set_xlim([self.time.min(), self.time.max()])
        
        y = np.clip(self.timediff, a_min=0, a_max=np.percentile(self.timediff, 99.5))
        hist, bins, _ = ax[1].hist(y, bins=100, fill=False, histtype="step", density=True)
        x = np.linspace(0, bins.max(), 200)
        ax[1].plot(x, self.kappa * np.exp(- self.kappa * x), color="black")
        ax[1].set_yscale("log")
        ax[1].set_xlabel(r"Time Difference $\Delta t$", size=20)
        ax[1].set_xlim([0,None])

def estimate_event_rate(adata, key_time="Time"):
    """ Estimates overall event rate from time of events/cells in adata.obs[key_time], using EventRateEstimator.
    """
    return EventRateEstimator(adata.obs[key_time]).rate
