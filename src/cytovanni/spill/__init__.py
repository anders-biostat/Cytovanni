"""Single-stain spectra: collect and fit single-stain controls, compare spectra across measurements, build panels.

Typical workflow: `collect_bead_controls_regularflow` (or `collect_bead_controls_spectralflow`) reads and gates the
single-stain .fcs files of one measurement; the single-stain fitters (`LinearSingleStainSpectrumFit`,
`OutliersafeLinearSingleStainSpectrumFit`, `OutliersafeTorchSingleStainSpectrumFit`, `PosNegSingleStainSpectrumFit`)
fit one spectrum per dye and `collect_single_fits` combines them into one spectra matrix per fit method.
`SpilloverMeasurement` loads these matrices for one measurement (and calibrates them with the rainbow calibration of
that measurement), `SpilloverCollection` collects many measurements per dye and fits a PCA of the spectra of every
dye, and `PanelConfiguration` combines the chosen PCA components of the dyes of one panel into the panel embedding
used by `cytovanni.ref`.
"""
from .autospill import make_AutoSpill_fcs_config
from .utils import collect_bead_controls_regularflow, collect_bead_controls_spectralflow
from .stainfit import LinearSingleStainSpectrumFit, PosNegSingleStainSpectrumFit
from .stainfit import OutliersafeLinearSingleStainSpectrumFit, OutliersafeTorchSingleStainSpectrumFit
from .stainfit import collect_single_fits, collect_peakpos_fromposneg
from .collection import SpilloverMeasurement, SpilloverCollection
from .panel import PanelConfiguration
