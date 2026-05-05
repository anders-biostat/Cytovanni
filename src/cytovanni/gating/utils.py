import numpy as np

class CombinedGate():
    """ Simple wrapper to combine gates.
    """
    def __init__(self, gates):
        self.gates = gates
    
    def apply(self, ad):
        return np.vstack([gate.apply(ad) for gate in self.gates]).all(axis=0)
        