"""Rainbow bead calibration of detector channels across batches.

Workflow: read and scatter-gate every rainbow bead measurement (`cytovanni.io.readfcs_rainbow_sample`,
`cytovanni.gating.gate_scatter_bead_sample`), assign the beads to peaks with `fit_rainbow_GMM` (and
`get_rainbow_GMM_gate`), fit a `RainbowCalibrator` on all measurements, and calibrate samples with
`RainbowCalibrator.add_calibrated` (``layers['calibrated']`` = raw times the factors of the sample's batch).
`get_config_from_rainbow_reference` and `get_voltages_from_rainbow` help to set PMT voltages that reproduce a
reference measurement.
"""
from .utils import scale_spectra_wfactors, extract_logfactor_laserfactor, extract_logfactor_laserfactor_bylaser, reorder_peaks

from .gmm import fit_rainbow_GMM, get_rainbow_GMM_gate

from .plot import plot_rainbow_channel_GMM_single, plot_GMM_llh, plot_eval_RBInt_fit

from .setvolt import get_config_from_rainbow_reference
from .setvolt import get_voltages_from_rainbow, color_voltageconfig

from .rboe import RainbowBatchOrdinalEncoder

from .rbim_scatter import RainbowScatterCalibrationModule
from .rbim_data import RainbowFluorescenceGMMCalibrationDataset
from .rbim_fluorescence import RainbowFluorescenceGMMCalibrationModule, RainbowFluorescenceGMMCalibrationTrainer
from .rbim import RainbowCalibrator

from .pmt import PMTExponentFitter
