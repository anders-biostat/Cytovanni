import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter

def pick_smp_cls(model):
    """ Get appropriate class for saved scale multiplier from name.
    """
    if model=="default":
        return DefaultMultiplier
    elif model=="unmxerrexclusion":
        return UnmxErrorExclusionMultiplier
    else:
        raise NotImplementedError(f"Cannot find weighting model {model}!")

class ScaleMultiplier(Module):
    def __init__(self):
        super().__init__()

class DefaultMultiplier(ScaleMultiplier):
    """ Default scale multiplier.
    """
    NAME = "default"
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def export(self):
        """ Export self to dictionary.
        """
        return {}
    
    @classmethod
    def from_exported(cls, exported, *args, **kwargs):
        """ Restore from exported.
        """
        return cls()
    
    def forward(self, x, scale):
        """ Take data that was standardised by transformer, as well as scale the standardised data should
            be multiplied with, applies simple multiplication.
        """
        return x * scale

class UnmxErrorExclusionMultiplier(ScaleMultiplier):
    """ Occasionally, the spreading error caused by specific dyes may be increased in some cases.
        This class can exclude those events from contributing to the gradient of the scaling factor
        of some markers, to ensure that it is trained on the positive cells, not on the spreading error.
    """
    NAME = "unmxerrexclusion"
    
    def __init__(self, transformer, groups=[]):
        """ 
            
            :params transformer: Module. The class that handles the spectral standardisation.
            
            :params groups: iterable. List of groups with the markers that should be excluded. Every entry should be (stain name, linear positive cutoff, stains to exclude). I.e. ('PE', 1e4, ['BUV396']) detaches the gradient of the scaling factor of 'BUV396' for all events where 'PE' is above 1e4.
        """
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
        """ Set requires_grad properly on all parameters.
        """
        self.group_stain_idx.requires_grad_(False)
        self.group_arange.requires_grad_(False)
        self.group_value.requires_grad_(False)
        self.block_mask.requires_grad_(False)
    
    def forward(self, xc, scale):
        """ Take data that was standardised by transformer, return weighting for every event.
            
            Don't bother with normalization of weights, MMD loss class anyway enforces proper normalization.
        """
        if self.is_dummy:
            return xc * scale
        else:
            mask = xc.detach()[:,self.group_stain_idx] > self.group_threshold[self.group_arange,self.group_stain_idx][None]
            detach_mask = torch.clamp((mask.to(torch.float32) @ self.block_mask), 0., 1.)
            return xc * (detach_mask * scale.detach() + (1-detach_mask) * scale)
    
    def export(self):
        """ Export self to dictionary.
        """
        return {"groups":self.groups}
    
    @classmethod
    def from_exported(cls, exported, transformer):
        """ Restore from exported.
        """
        return cls(transformer, groups=exported["groups"])



