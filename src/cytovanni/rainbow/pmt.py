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
    
    def fit_single(self, channel, plot=False):
        logshift, voltage, intensity = self.df_logshift[channel], self.df_voltage[channel], self.get_intensity(channel)
        mask = (logshift==logshift) & (voltage>0) & (intensity>0)
        logshift, voltage, intensity = logshift[mask], voltage[mask], intensity[mask]
        xdata = np.log(np.stack([voltage, intensity]))
        ydata = logshift
        
        if np.unique(voltage).shape[0]<=1:
            printwtime(f"Not enough different values to fit exponent for channel {channel}!")
            return [0.,1.]

        def fitfunct(xdata, intercept, slope_PMT, slope_LAS):
            """ param [intercept, slope_PMT, slope_LAS]
            """
            return xdata[0] * slope_PMT + intercept + xdata[1] * (slope_LAS if self.fit_laser_exponent else 1.)
        def combinexfct(xdata, intercept, slope_PMT, slope_LAS):
            return xdata[0] * slope_PMT + xdata[1] * slope_LAS

        params = scipy.optimize.curve_fit(fitfunct, xdata, ydata, p0=[0,10,1])[0]

        if plot:
            sns.regplot(x=combinexfct(xdata, *params), y=ydata)
            plt.xlabel("logshift")
            plt.show()

        return params
    
    def fit_all(self):
        params = [self.fit_single(channel) for channel in self.channels]
        self.parameters = pd.DataFrame(np.asarray(params), columns=["intercept","exponent_PMT","exponent_LAS"], index=self.channels)
    
    def plot_exponents(self):
        sns.scatterplot(x=self.parameters["exponent"], y=self.parameters["exponent"].index)
        plt.xlabel("PMT Scaling Exponent")
        plt.ylabel("Detector Channel")
    
    def plot_fit(self):
        def combinexfct(xdata, intercept, slope_PMT, slope_LAS):
            return xdata[0] * slope_PMT + xdata[1] * slope_LAS
        def plot_single(ax, channel):
            logshift, voltage, intensity = self.df_logshift[channel], self.df_voltage[channel], self.get_intensity(channel)
            mask = (logshift==logshift) & (voltage>0) & (intensity>0)
            logshift, voltage, intensity = logshift[mask], voltage[mask], intensity[mask]
            xdata = np.log(np.stack([voltage, intensity]))
            ydata = logshift
            
            x, y = combinexfct(xdata, *self.parameters.loc[channel]), ydata
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
        def combinexfct(xdata, intercept, slope):
            return xdata[0] + xdata[1] / slope
        def get_single(channel):
            logshift, voltage, intensity = self.df_logshift[channel], self.df_voltage[channel], self.get_intensity(channel)
            #mask = (logshift==logshift) & (voltage>0) & (intensity>0)
            #logshift, voltage, intensity = logshift[mask], voltage[mask], intensity[mask]
            xdata = silent_log(np.stack([voltage, intensity]))

            x = combinexfct(xdata, *self.parameters.loc[channel])
            return x

        theoshift = np.asarray([get_single(c) for c in self.channels]).T
        theoshift[np.isinf(theoshift)] = np.nan
        return pd.DataFrame(theoshift, columns=self.channels, index=self.df_logshift.index)
    
    def write(self, path):
        self.parameters.to_csv(path)