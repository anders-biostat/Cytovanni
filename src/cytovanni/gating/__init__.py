"""Gates for AnnData and DataFrames, and automatic singlet gating of bead samples.

Manual gates (`PolygonGate`, `LineGate`, `MinMaxGate`, `FluorescenceMinMaxGate`, `CutoffUnmxGate`, `UnmxMinGate`,
`CombinedGate`) return a boolean mask from ``gate.apply(adata)`` and ``gate.apply_df(df)`` (True: event kept); they
do not change the data unless an `addkey` is given. `gate_scatter_bead_sample` gates bead samples on singlets
automatically.
"""
from .utils import CombinedGate

from .polygon import PolygonGate
from .minmax import MinMaxGate, FluorescenceMinMaxGate
from .unmx import CutoffUnmxGate, UnmxMinGate
from .line import LineGate

from .autospill import AutoSpillTesselationGating

from .bead import gate_scatter_bead_sample
