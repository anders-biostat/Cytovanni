"""Batch standardisation with reference / overlap samples.

The overlap fit (`OverlapFitModel`) fits per-batch dye spectra (`PanelEmbeddingModule`) and stain scale factors
(`SpectralFitTransformer`, `ScalingTransformer`) on samples measured in several batches (`SampleOverlapDataset`,
batch labels encoded by `MultiOrdinalEncoder`). The functions of `cytovanni.ref.factor` fit per-batch scale
factors directly, without a full model.
"""
from .panel import PanelModule, PanelEmbeddingModule, FixedPanelModule
from .moe import MultiOrdinalEncoder
from .data import SampleOverlapDataset, adatas_process_anchors

from .model_weight import DummyWeights, RareMarkerWeights
from .model_transformer_smp import DefaultMultiplier, UnmxErrorExclusionMultiplier
from .model_transformer import DummyTransformer, SpectralFitTransformer, ScalingTransformer
from .model import OverlapFitModel

from .factor import fit_factor_Sinkhorn, fit_factor_Sinkhorn_grouped, fit_factor_MeanMatching, fit_factor_PercentileMatching
from .factor import fit_factors_Sinkhorn

from .plot import plot_eval_histogram, plot_eval_NxN
