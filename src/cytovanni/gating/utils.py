import numpy as np

class CombinedGate():
    """ Simple wrapper to combine gates.
    """
    def __init__(self, gates):
        self.gates = gates
    
    def apply(self, ad):
        return np.vstack([gate.apply(ad) for gate in self.gates]).all(axis=0)

    def apply_df(self, df):
        """ Apply all gates to a DataFrame (see the apply_df of the individual gates), True where all keep the event. """
        return np.vstack([gate.apply_df(df) for gate in self.gates]).all(axis=0)
        