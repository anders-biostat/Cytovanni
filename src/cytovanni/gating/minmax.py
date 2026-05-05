import numpy as np

class MinMaxGate():
    """ Gate AnnData to within a range across all given channels.
    """
    def __init__(self, channels, v_min, v_max, layer="raw"):
        self.channels = list(channels)
        self.v_min, self.v_max = v_min, v_max
        self.layer = layer
    
    def __repr__(self):
        repstr = f"MinMax gate within [{self.v_min}, {self.v_max}], gate on channels {self.channels}."
        return repstr
    
    def apply(self, ad, addkey=""):
        vals = ad[:,self.channels].layers[self.layer]
        mask = np.all((vals>=self.v_min) & (vals <=self.v_max), axis=1)
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        vals = df[self.channels].to_numpy()
        mask = np.all((vals>=self.v_min) & (vals <=self.v_max), axis=1)
        return mask

class FluorescenceMinMaxGate(MinMaxGate):
    """ MinMaxGate, uses all fluorescence channels in cytoconfig.
    """
    def __init__(self, cytoconfig, v_min=-1e3, max_tolerance=1.5, layer="raw", v_max=None):
        super().__init__(cytoconfig.channels_fluorescence, v_min, cytoconfig.max_linear_range*max_tolerance if v_max is None else v_max)
    
