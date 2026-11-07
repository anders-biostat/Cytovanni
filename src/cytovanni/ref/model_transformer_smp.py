"""Scale multipliers: apply the stain scale factors in a transformer, optionally with blocked gradients."""
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter

def pick_smp_cls(model):
    """Return the scale multiplier class for its saved name (``NAME``).

    Parameters
    ----------
    model : {'default', 'unmxerrexclusion'}
        Name stored in an exported transformer (``smp_name``).

    Returns
    -------
    type
        `DefaultMultiplier` or `UnmxErrorExclusionMultiplier`.

    Raises
    ------
    NotImplementedError
        For an unknown name (custom classes are passed as `cls_smp` when loading).
    """
    if model=="default":
        return DefaultMultiplier
    elif model=="unmxerrexclusion":
        return UnmxErrorExclusionMultiplier
    else:
        raise NotImplementedError(f"Cannot find weighting model {model}!")

class ScaleMultiplier(Module):
    """Base class of the scale multipliers: apply the stain scale factors to the unmixed data in a transformer.

    Subclasses define ``NAME``, ``forward(xc, scale)``, ``export()`` and ``from_exported(exported, transformer)``. The
    forward value is always ``xc * scale``; subclasses only change which gradients reach the scale factors.
    """
    def __init__(self):
        super().__init__()

class DefaultMultiplier(ScaleMultiplier):
    """Plain multiplication of the unmixed data with the stain scale factors; the default of the transformers.

    Parameters
    ----------
    *args, **kwargs
        Ignored.
    """
    NAME = "default"
    
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
    def from_exported(cls, exported, *args, **kwargs):
        """Restore from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export` (ignored).
        *args, **kwargs
            Ignored (the transformer).

        Returns
        -------
        DefaultMultiplier
        """
        return cls()
    
    def forward(self, x, scale):
        """Multiply the unmixed data by the stain scale factors.

        Parameters
        ----------
        x : torch.Tensor of shape (events, stain)
            Unmixed data (linear scale).
        scale : torch.Tensor of shape (events, stain) or broadcastable
            Stain scale factors of the events' batches.

        Returns
        -------
        torch.Tensor of shape (events, stain)
            ``x * scale``.
        """
        return x * scale

class UnmxErrorExclusionMultiplier(ScaleMultiplier):
    """Exclude events with a large spreading error from the gradient of selected stain scale factors.

    The spreading error caused by some dyes can be increased in some batches. For events above a cutoff in such a
    stain, the gradient of the scale factors of the affected stains is blocked, so these factors are trained on the
    other events (e.g. the truly positive cells) and not on the spreading error. The forward value is unchanged
    (``xc * scale``), and the gradient with respect to the unmixed data itself (the spectra) is not affected.

    Set it on a transformer after creating it: ``transformer.set_smp(UnmxErrorExclusionMultiplier(transformer,
    groups))``.

    Parameters
    ----------
    transformer : torch.nn.Module
        The transformer (`SpectralFitTransformer` or `ScalingTransformer`); its ``stains`` define the stain names.
    groups : list of tuple, default []
        ``(stain, cutoff, excluded_stains)`` per group, e.g. ``('PE', 1e4, ['BUV396'])`` blocks the gradient of the
        BUV396 scale factor for all events with PE above 1e4. Names are from ``transformer.stains``; `cutoff` is
        on the linear scale of the unmixed data before the stain scaling (after the channel scaling of a
        `SpectralFitTransformer`). An empty list gives a plain multiplication.
    """
    NAME = "unmxerrexclusion"
    
    def __init__(self, transformer, groups=[]):
        super().__init__()
        self.groups = groups
        self.is_dummy = len(groups)==0
        if not self.is_dummy:
            
            self.group_stain_idx = Parameter(torch.tensor([list(transformer.stains).index(c[0]) for c in self.groups], dtype=torch.long), requires_grad=False)
            self.group_arange = Parameter(torch.arange(len(self.groups), dtype=torch.long), requires_grad=False)
            group_threshold = torch.zeros((len(self.groups), len(transformer.stains)), dtype=torch.float32)
            group_threshold[self.group_arange, self.group_stain_idx] = torch.tensor([c[1] for c in self.groups], dtype=torch.float32)
            self.group_threshold = Parameter(group_threshold, requires_grad=False)
            
            block_mask = torch.zeros((len(self.groups), len(transformer.stains)), dtype=torch.float32)
            for i in range(len(self.groups)):
                block_mask[i, [list(transformer.stains).index(c) for c in self.groups[i][2]]] = 1.
            self.block_mask = Parameter(block_mask, requires_grad=False)
    
    def requires_grad_(self, requires_grad=False):
        """Keep the group definitions (stored as parameters) non-trainable, whatever `requires_grad` is.

        Parameters
        ----------
        requires_grad : bool, default False
            Ignored: the group definitions are never trained.

        Returns
        -------
        UnmxErrorExclusionMultiplier
            self.
        """
        if self.is_dummy: return self
        self.group_stain_idx.requires_grad_(False)
        self.group_arange.requires_grad_(False)
        self.group_threshold.requires_grad_(False)
        self.block_mask.requires_grad_(False)
        return self
    
    def forward(self, xc, scale):
        """Multiply the unmixed data by the stain scale factors, with the gradient blocked for the excluded events.

        Parameters
        ----------
        xc : torch.Tensor of shape (events, stain)
            Unmixed data (linear scale, before the stain scaling); compared with the group cutoffs.
        scale : torch.Tensor of shape (events, stain)
            Stain scale factors of the events' batches.

        Returns
        -------
        torch.Tensor of shape (events, stain)
            ``xc * scale``; for events above a group's cutoff, the gradient with respect to `scale` is detached in
            that group's excluded stains.
        """
        if self.is_dummy:
            return xc * scale
        else:
            mask = xc.detach()[:,self.group_stain_idx] > self.group_threshold[self.group_arange,self.group_stain_idx][None]
            detach_mask = torch.clamp((mask.to(self.block_mask.dtype) @ self.block_mask), 0., 1.)
            return xc * (detach_mask * scale.detach() + (1-detach_mask) * scale)
    
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
            The restored transformer.

        Returns
        -------
        UnmxErrorExclusionMultiplier
        """
        return cls(transformer, groups=exported["groups"])



