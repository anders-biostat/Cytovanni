"""Event rate of a recording, estimated from the event times."""
import numpy as np
import matplotlib.pyplot as plt

class EventRateEstimator():
    r"""Estimate the overall event rate of a recording from its event times.

    Events are modelled as a Poisson point process (with a maximal rate, e.g. due to doublet generation), so the time
    differences between consecutive events follow an exponential distribution
    :math:`f(\Delta t;\kappa) = \kappa \exp(-\kappa \Delta t)`. Its maximum-likelihood estimate would be
    :math:`\kappa = N / \sum_i^N \Delta t_i`, i.e. one over the mean time difference, which is sensitive to outliers
    (e.g. pauses in the acquisition). Since the median of the distribution is :math:`\ln(2)/\kappa`, the class uses the
    more robust estimator :math:`\kappa = \ln(2) / \operatorname{median}(\Delta t_i)` instead.

    Parameters
    ----------
    times : array-like of float
        Time of every event, e.g. ``adata.obs['Time']``, in any order (sorted internally). The rate is per unit of
        these times, which this class does not convert: the ``Time`` channel of an .fcs file is usually stored in
        units of the instrument's time step ($TIMESTEP keyword), not in seconds.

    Attributes
    ----------
    time : numpy.ndarray
        Sorted event times.
    timediff : numpy.ndarray
        Differences between consecutive sorted event times (length N - 1).
    N : int
        Number of events.
    kappa : float
        Estimated rate, ``ln(2) / median(timediff)``, in events per unit of `times`.
    rate : float
        Same as `kappa`.

    Examples
    --------
    >>> t = np.cumsum(np.random.exponential(1 / 50., size=20000))
    >>> EventRateEstimator(t).rate  # about 50
    """
    def __init__(self, times):
        self.time = np.sort(np.asarray(times))
        self.timediff = self.time[1:] - self.time[:-1]
        self.N = self.time.shape[0]
        self.kappa = np.log(2) / np.median(self.timediff)
        self.rate = self.kappa

    def plot(self):
        """Plot the event times and the time differences against the fitted rate.

        Left: histogram of the event times (200 bins, counts) with the expected count per bin at the estimated rate;
        dips show interruptions or rate changes during the recording. Right: density of the time differences (clipped
        at their 99.5th percentile, log y axis) with the fitted exponential density. Creates a new figure and does not
        call ``plt.show()``.
        """
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
    """Estimate the overall event rate of a sample from its event times.

    Parameters
    ----------
    adata : anndata.AnnData
        Sample with the event times in ``adata.obs[key_time]``.
    key_time : str, default "Time"
        Column of ``adata.obs`` holding the event times.

    Returns
    -------
    float
        Events per unit of ``adata.obs[key_time]`` (instrument time unit, see `EventRateEstimator`).

    See Also
    --------
    EventRateEstimator : the estimator and a diagnostic plot.
    """
    return EventRateEstimator(adata.obs[key_time]).rate
