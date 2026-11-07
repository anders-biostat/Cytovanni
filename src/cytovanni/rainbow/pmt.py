"""Fit how the detector gain depends on the PMT voltage (and laser power) from rainbow calibration results."""
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import printwtime, silent_log, unrolled_subplots

class PMTExponentFitter():
    """Fit a power law of the detector gain in the PMT voltage (and laser power), per channel.

    For every channel, fits ``logshift = intercept + exponent_PMT * ln(voltage) + exponent_LAS * ln(intensity)`` by
    least squares (``scipy.optimize.curve_fit``) over the measurements, i.e. the gain is proportional to
    ``voltage ** exponent_PMT * intensity ** exponent_LAS``. Use it on measurements of the same beads at different
    voltages (and laser powers), e.g. with the log gains of a `RainbowCalibrator` fit, to estimate the PMT exponents
    used by `get_config_from_rainbow_reference`.

    Parameters
    ----------
    df_logshift : pandas.DataFrame
        Natural log of the gain, shape (measurement, channel), e.g. ``-numpy.log(rbint.get_calibration_factors_all())``.
        NaN entries are left out of the fit. Its columns are the fitted channels.
    df_voltage : pandas.DataFrame
        PMT voltage, same index and columns as `df_logshift` (e.g. ``adata.var['pnv']`` per measurement). Rows with
        voltage <= 0 are left out.
    df_intensity : pandas.DataFrame, optional
        Laser power of the channel's laser, same index and columns (e.g. ``adata.var['laser_wattage']``); rows with
        intensity <= 0 are left out. Default: intensity 1 everywhere (no laser term); then use
        ``fit_laser_exponent=False``, the laser exponent cannot be determined.
    fit_laser_exponent : bool, default True
        Fit `exponent_LAS`; if False it is fixed to 1.

    Attributes
    ----------
    channels : list of str
        Columns of `df_logshift`.
    parameters : pandas.DataFrame
        Set by `fit_all`: index channel, columns 'intercept', 'exponent_PMT', 'exponent_LAS'.

    Examples
    --------
    >>> fitter = PMTExponentFitter(-np.log(factors), df_voltage, df_intensity, fit_laser_exponent=True)
    >>> fitter.fit_all()
    >>> fitter.parameters["exponent_PMT"]
    """
    def __init__(self, df_logshift, df_voltage, df_intensity=None, fit_laser_exponent=True):
        self.df_logshift, self.df_voltage, self.df_intensity = df_logshift, df_voltage, df_intensity
        self.channels = list(self.df_logshift.columns)
        self.fit_laser_exponent = fit_laser_exponent
    
    def get_intensity(self, channel):
        """Laser intensity of one channel per measurement: the column of `df_intensity`, or ones without it.

        Parameters
        ----------
        channel : str
            Column of `df_logshift`.

        Returns
        -------
        pandas.Series or numpy.ndarray
            Intensity per measurement.
        """
        if self.df_intensity is not None:
            return self.df_intensity[channel]
        else:
            return np.ones_like(self.df_voltage[channel])
    
    def _xdata(self, channel, drop_invalid=True):
        """Log voltage and log intensity (shape (2, N)) and logshift of channel, optionally restricted to valid rows."""
        logshift, voltage, intensity = self.df_logshift[channel], self.df_voltage[channel], self.get_intensity(channel)
        if drop_invalid:
            mask = (logshift==logshift) & (voltage>0) & (intensity>0)
            logshift, voltage, intensity = logshift[mask], voltage[mask], intensity[mask]
            return np.log(np.stack([voltage, intensity])), logshift
        return silent_log(np.stack([voltage, intensity])), logshift
    
    def _slope_LAS(self, slope_LAS):
        """Laser exponent used by the model: fitted, or fixed to 1 if not fit_laser_exponent."""
        return slope_LAS if self.fit_laser_exponent else 1.
    
    def _combinex(self, xdata, intercept, slope_PMT, slope_LAS):
        """Model without intercept: log voltage * PMT exponent + log intensity * laser exponent."""
        return xdata[0] * slope_PMT + xdata[1] * self._slope_LAS(slope_LAS)
    
    def fit_single(self, channel, plot=False):
        """Fit the model of the class docstring for one channel.

        Parameters
        ----------
        channel : str
            Column of `df_logshift`.
        plot : bool, default False
            Plot the data against the fitted model (``plt.show()``).

        Returns
        -------
        numpy.ndarray or list of float
            ``[intercept, exponent_PMT, exponent_LAS]``; exponent_LAS is 1 if not `fit_laser_exponent`. All NaN (with a
            printed message) if the valid rows have fewer than two distinct voltages.

        Notes
        -----
        Start values: intercept 0, PMT exponent 10, laser exponent 1. With `fit_laser_exponent`, ``curve_fit`` needs
        at least three valid rows (it raises a TypeError otherwise).
        """
        xdata, ydata = self._xdata(channel)
        
        if np.unique(xdata[0]).shape[0]<=1:
            printwtime(f"Not enough different values to fit exponent for channel {channel}!")
            return [np.nan, np.nan, np.nan]

        if self.fit_laser_exponent:
            def fitfunct(xdata, intercept, slope_PMT, slope_LAS):
                return intercept + self._combinex(xdata, intercept, slope_PMT, slope_LAS)
            params = scipy.optimize.curve_fit(fitfunct, xdata, ydata, p0=[0,10,1])[0]
        else: # laser exponent fixed at 1: only intercept and PMT exponent are free (two voltages suffice)
            def fitfunct(xdata, intercept, slope_PMT):
                return intercept + self._combinex(xdata, intercept, slope_PMT, 1.)
            params = np.append(scipy.optimize.curve_fit(fitfunct, xdata, ydata, p0=[0,10])[0], 1.)

        if plot:
            sns.regplot(x=self._combinex(xdata, *params), y=ydata)
            plt.xlabel("logshift")
            plt.show()

        return params
    
    def fit_all(self):
        """Fit every channel with `fit_single` and store the result in `parameters` (index channel)."""
        params = [self.fit_single(channel) for channel in self.channels]
        self.parameters = pd.DataFrame(np.asarray(params), columns=["intercept","exponent_PMT","exponent_LAS"], index=self.channels)
    
    def plot_exponents(self):
        """Scatter plot of the fitted PMT exponent per channel (after `fit_all`)."""
        sns.scatterplot(x=self.parameters["exponent_PMT"], y=self.parameters["exponent_PMT"].index)
        plt.xlabel("PMT Scaling Exponent")
        plt.ylabel("Detector Channel")
    
    def plot_fit(self):
        """Plot, per channel, the measured log gain against the fitted model term (without intercept), with r².

        One panel per channel (after `fit_all`): x is ``exponent_PMT * ln(voltage) + exponent_LAS * ln(intensity)``,
        y the measured logshift, with a regression line (seaborn ``regplot``) and the squared correlation.
        """
        def plot_single(ax, channel):
            xdata, ydata = self._xdata(channel)
            x, y = self._combinex(xdata, *self.parameters.loc[channel]), ydata
            sns.regplot(x=x, y=y, ax=ax)
            
            r2 = np.corrcoef(x,y)[0,1]**2
            ax.text(.1,.9, f"r²={r2:.4f}", transform=ax.transAxes)
            
            ax.set_title(channel)
            ax.set_xlabel("Signal Logshift Theoretical")
            ax.set_ylabel("Signal Logshift Fit")
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig, ax = unrolled_subplots(len(self.channels), elsize=(4,4))
        for ax_, channel in zip(ax, self.channels):
            plot_single(ax_, channel)
    
    def get_theoretical_shift(self):
        """Log gain predicted by the fitted model for every measurement and channel (after `fit_all`).

        Returns
        -------
        pandas.DataFrame
            Same index and columns as `df_logshift`, intercept included; NaN where the voltage or intensity is not
            positive.
        """
        def get_single(channel):
            xdata, _ = self._xdata(channel, drop_invalid=False)
            intercept = self.parameters.loc[channel, "intercept"]
            return intercept + self._combinex(xdata, *self.parameters.loc[channel])

        theoshift = np.asarray([get_single(c) for c in self.channels]).T
        theoshift[np.isinf(theoshift)] = np.nan
        return pd.DataFrame(theoshift, columns=self.channels, index=self.df_logshift.index)
    
    def write(self, path):
        """Write `parameters` (after `fit_all`) to a .csv file.

        Parameters
        ----------
        path : str
            Output path.
        """
        self.parameters.to_csv(path)