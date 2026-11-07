import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import printwtime, silent_log, unrolled_subplots

class PMTExponentFitter():
    def __init__(self, df_logshift, df_voltage, df_intensity=None, fit_laser_exponent=True):
        self.df_logshift, self.df_voltage, self.df_intensity = df_logshift, df_voltage, df_intensity
        self.channels = list(self.df_logshift.columns)
        self.fit_laser_exponent = fit_laser_exponent
    
    def get_intensity(self, channel):
        if self.df_intensity is not None:
            return self.df_intensity[channel]
        else:
            return np.ones_like(self.df_voltage[channel])
    
    def _xdata(self, channel, drop_invalid=True):
        """ Log voltage and log laser intensity (shape (2, N)) and logshift of channel, optionally restricted to valid rows.
        """
        logshift, voltage, intensity = self.df_logshift[channel], self.df_voltage[channel], self.get_intensity(channel)
        if drop_invalid:
            mask = (logshift==logshift) & (voltage>0) & (intensity>0)
            logshift, voltage, intensity = logshift[mask], voltage[mask], intensity[mask]
            return np.log(np.stack([voltage, intensity])), logshift
        return silent_log(np.stack([voltage, intensity])), logshift
    
    def _slope_LAS(self, slope_LAS):
        """ Laser exponent used by the model: fitted, or fixed to 1 if not fit_laser_exponent.
        """
        return slope_LAS if self.fit_laser_exponent else 1.
    
    def _combinex(self, xdata, intercept, slope_PMT, slope_LAS):
        """ Model without intercept: log voltage * PMT exponent + log intensity * laser exponent.
        """
        return xdata[0] * slope_PMT + xdata[1] * self._slope_LAS(slope_LAS)
    
    def fit_single(self, channel, plot=False):
        """ Fit logshift = intercept + exponent_PMT * log(voltage) + exponent_LAS * log(intensity) for one channel.
            Returns [intercept, exponent_PMT, exponent_LAS], all nan if the channel has fewer than two distinct voltages.
            exponent_LAS stays at its start value 1 if not fit_laser_exponent.
        """
        xdata, ydata = self._xdata(channel)
        
        if np.unique(xdata[0]).shape[0]<=1:
            printwtime(f"Not enough different values to fit exponent for channel {channel}!")
            return [np.nan, np.nan, np.nan]

        def fitfunct(xdata, intercept, slope_PMT, slope_LAS):
            return intercept + self._combinex(xdata, intercept, slope_PMT, slope_LAS)

        params = scipy.optimize.curve_fit(fitfunct, xdata, ydata, p0=[0,10,1])[0]
        if not self.fit_laser_exponent:
            params[2] = 1.

        if plot:
            sns.regplot(x=self._combinex(xdata, *params), y=ydata)
            plt.xlabel("logshift")
            plt.show()

        return params
    
    def fit_all(self):
        params = [self.fit_single(channel) for channel in self.channels]
        self.parameters = pd.DataFrame(np.asarray(params), columns=["intercept","exponent_PMT","exponent_LAS"], index=self.channels)
    
    def plot_exponents(self):
        sns.scatterplot(x=self.parameters["exponent_PMT"], y=self.parameters["exponent_PMT"].index)
        plt.xlabel("PMT Scaling Exponent")
        plt.ylabel("Detector Channel")
    
    def plot_fit(self):
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
        """ Logshift predicted by the fitted model (intercept included) for every sample and channel, shape of df_logshift.
            nan where voltage or intensity is not positive.
        """
        def get_single(channel):
            xdata, _ = self._xdata(channel, drop_invalid=False)
            intercept = self.parameters.loc[channel, "intercept"]
            return intercept + self._combinex(xdata, *self.parameters.loc[channel])

        theoshift = np.asarray([get_single(c) for c in self.channels]).T
        theoshift[np.isinf(theoshift)] = np.nan
        return pd.DataFrame(theoshift, columns=self.channels, index=self.df_logshift.index)
    
    def write(self, path):
        self.parameters.to_csv(path)