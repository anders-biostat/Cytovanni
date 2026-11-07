"""Per-event weightings of the overlap loss (marker weights)."""
import warnings
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter

def pick_weight_cls(model):
    """Return the marker weighting class for its saved name (``NAME``).

    Parameters
    ----------
    model : {'dummy', 'raremarker'}
        Name stored in an exported model (``markerweight_name``).

    Returns
    -------
    type
        `DummyWeights` or `RareMarkerWeights`.

    Raises
    ------
    NotImplementedError
        For an unknown name (custom classes are passed to `OverlapFitModel.from_saved` directly).
    """
    if model=="dummy":
        return DummyWeights
    elif model=="raremarker":
        return RareMarkerWeights
    else:
        raise NotImplementedError(f"Cannot find weighting model {model}!")

class DummyWeights(Module):
    """No event weighting (all events weigh the same); the default marker weighting of `OverlapFitModel`.

    Also shows the interface of a marker weighting: ``NAME``, `export`, `from_exported` and ``forward(x, idx_batch,
    samp_cov)`` returning per-event weights or None.

    Parameters
    ----------
    *args, **kwargs
        Ignored.
    """
    NAME = "dummy"
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def export(self):
        """Export to a dict (empty, nothing to store).

        Returns
        -------
        dict
        """
        return {}
    
    @classmethod
    def from_exported(cls, exported, transformer, *args, **kwargs):
        """Restore from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export` (ignored).
        transformer : torch.nn.Module
            The model's transformer (ignored).
        *args, **kwargs
            Ignored.

        Returns
        -------
        DummyWeights
        """
        return cls()
    
    def forward(self, x, idx_batch, samp_cov):
        """Return None (uniform weights).

        Parameters
        ----------
        x : torch.Tensor of shape (events, stain)
            Transformer output (ignored).
        idx_batch : dict of str to torch.Tensor
            Batch indices (ignored).
        samp_cov : dict
            Sample covariates (ignored).

        Returns
        -------
        None
        """
        return None

class RareMarkerWeights(Module):
    """Upweight the events positive for rare markers in the overlap loss.

    Every group is a stain with a positivity cutoff and a weight: events above the cutoff get that weight, the others
    weight 1; an event positive in several groups gets the largest of their weights. The weights are normalised to
    mean 1 per sample. Only a +/- split per marker is used; choosing a reference sample with enough positive cells
    is probably the better fix.

    Parameters
    ----------
    transformer : torch.nn.Module
        The model's transformer (`SpectralFitTransformer` or `ScalingTransformer`); its ``stains`` define the stain
        names and its ``apply_transformation`` the scale of the cutoffs.
    groups : list of tuple, default []
        ``(stain, cutoff, weight)`` per group, e.g. ``('PE', 1e4, 2)`` doubles the weight of every event with PE
        above 1e4. `stain` is a name in ``transformer.stains`` (dye names for a panel embedding, the index of the
        spectra for a `ScalingTransformer`). `cutoff` is on the linear scale of the standardised, unmixed data
        during the fit: for a `SpectralFitTransformer` with anchored batches the units of the anchor batches (their
        factors stay fixed), for a `ScalingTransformer` the scale with log factors of mean 0 over batches. Weights
        below 1 act as 1. An empty list gives no weighting (`forward` returns None).
    """
    NAME = "raremarker"
    
    def __init__(self, transformer, groups=[]):
        super().__init__()
        self.groups = groups
        self.is_dummy = len(groups)==0
        if not self.is_dummy:
            
            self.group_stain_idx = Parameter(torch.tensor([list(transformer.stains).index(c[0]) for c in groups], dtype=torch.long), requires_grad=False)
            self.group_arange = Parameter(torch.arange(len(groups), dtype=torch.long), requires_grad=False)
            self.group_weight = Parameter(torch.tensor([c[2] for c in groups], dtype=torch.float32), requires_grad=False)
            group_value = torch.zeros((len(groups), len(transformer.stains)), dtype=torch.float32)
            group_value[self.group_arange, self.group_stain_idx] = torch.tensor([c[1] for c in groups], dtype=torch.float32)
            self.group_value = Parameter(group_value, requires_grad=False)
            # store transformation this way to not register transformation parameters
            self.fct_transformation = transformer.apply_transformation
    
    def requires_grad_(self, requires_grad=False):
        """Keep the group definitions (stored as parameters) non-trainable, whatever `requires_grad` is.

        Parameters
        ----------
        requires_grad : bool, default False
            Ignored: the group definitions are never trained.

        Returns
        -------
        RareMarkerWeights
            self.
        """
        if self.is_dummy: return self
        self.group_stain_idx.requires_grad_(False)
        self.group_arange.requires_grad_(False)
        self.group_weight.requires_grad_(False)
        self.group_value.requires_grad_(False)
        return self
    
    def forward(self, x, idx_batch, samp_cov):
        """Return the weight of every event.

        Parameters
        ----------
        x : torch.Tensor of shape (events, stain)
            Transformer output (standardised, unmixed, transformed with ``transformer.apply_transformation``); the
            cutoffs are transformed the same way before the comparison.
        idx_batch : dict of str to torch.Tensor
            Batch indices (not used).
        samp_cov : dict
            Sample covariates (not used).

        Returns
        -------
        torch.Tensor of shape (events,) or None
            Weights with mean 1; None without groups.
        """
        if self.is_dummy: return None
        
        group_threshold = self.fct_transformation(self.group_value)
        mask = x.detach()[:,self.group_stain_idx] > group_threshold[self.group_arange,self.group_stain_idx][None]
        w = torch.clamp(mask.to(torch.float) * self.group_weight[None], min=1).max(dim=-1).values
        
        w = w*w.shape[0]/w.sum()
        
        return w
    
    def export(self):
        """Export to a dict (the groups).

        Returns
        -------
        dict
        """
        return {"groups":self.groups}
    
    @classmethod
    def from_exported(cls, exported, transformer):
        """Restore from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export`.
        transformer : torch.nn.Module
            The restored transformer of the model.

        Returns
        -------
        RareMarkerWeights
        """
        return cls(transformer, groups=exported["groups"])



