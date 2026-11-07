"""Combination of several gates."""
import numpy as np

class CombinedGate():
    """Gate that keeps the events kept by all of its gates (logical AND).

    Parameters
    ----------
    gates : list
        Gate objects (e.g. `PolygonGate`, `MinMaxGate`). `apply` calls ``gate.apply(ad)`` of each, `apply_df`
        ``gate.apply_df(df)``.

    Examples
    --------
    >>> gate_singlets = CombinedGate([gate_nd_SSC, gate_nd_FSC])
    >>> adata = adata[gate_singlets.apply(adata)]
    """
    def __init__(self, gates):
        self.gates = gates
    
    def apply(self, ad):
        """Apply all gates to an AnnData.

        Parameters
        ----------
        ad : anndata.AnnData
            Data as needed by the individual gates.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True where all gates keep the event.
        """
        return np.vstack([gate.apply(ad) for gate in self.gates]).all(axis=0)

    def apply_df(self, df):
        """Apply all gates to a DataFrame (see the `apply_df` of the individual gates).

        Parameters
        ----------
        df : pandas.DataFrame
            Events as needed by the individual gates.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True where all gates keep the event.
        """
        return np.vstack([gate.apply_df(df) for gate in self.gates]).all(axis=0)
        