import warnings
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter

def pick_weight_cls(model):
    """ Get appropriate class for saved transformers from name.
    """
    if model=="dummy":
        return DummyWeights
    elif model=="raremarker":
        return RareMarkerWeights
    else:
        raise NotImplementedError(f"Cannot find weighting model {model}!")

class DummyWeights(Module):
    """ Dummy weighting to showcase necessary parts.
    """
    NAME = "dummy"
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def export(self):
        """ Export self to dictionary.
        """
        return {}
    
    @classmethod
    def from_exported(cls, exported, transformer, *args, **kwargs):
        """ Restore from exported.
        """
        return cls()
    
    def forward(self, x, idx_batch, samp_cov):
        """ Take data that was standardised by transformer, return weighting for every event.
        """
        return None

class RareMarkerWeights(Module):
    """ Upweight rare celltype markers. Uses max weight for overlapping classes.
        
        Only distinguished markers as +/-, could maybe be improved.
        Probably better to just choose an appropriate reference.
    """
    NAME = "raremarker"
    
    def __init__(self, transformer, groups=[]):
        """ 
            
            :params transformer: Module. The class that handles the spectral standardisation.
            
            :params groups: iterable. List of groups to upweight. Every entry should be (stain name, linear positive cutoff, relative weight). I.e. ('PE', 1e4, 2) doubles the relative importance of every event where PE is above 1e4. Note that magnitude should be chosen to be appropriate for the anchors, as they will not get scaled. Group weights below one will get clipped to one.
        """
        super().__init__()
        self.groups = groups
        self.is_dummy = len(groups)==0
        if not self.is_dummy:
            
            self.group_stain_idx = Parameter(torch.tensor([list(transformer.stains).index(c[0]) for c in groups], dtype=torch.long), requires_grad=False)
            self.group_arange = Parameter(torch.arange(len(groups), dtype=torch.long), requires_grad=False)
            self.group_weight = Parameter(torch.tensor([c[2] for c in groups], dtype=torch.long), requires_grad=False)
            group_value = torch.zeros((len(groups), len(transformer.stains)), dtype=torch.float32)
            group_value[self.group_arange, self.group_stain_idx] = torch.tensor([c[1] for c in groups], dtype=torch.float32)
            self.group_value = Parameter(group_value, requires_grad=False)
            # store transformation this way to not register transformation parameters
            self.fct_transformation = transformer.apply_transformation
    
    def requires_grad_(self, requires_grad=False):
        """ Set requires_grad properly on all parameters.
        """
        self.group_stain_idx.requires_grad_(False)
        self.group_arange.requires_grad_(False)
        self.group_weight.requires_grad_(False)
        self.group_value.requires_grad_(False)
    
    def forward(self, x, idx_batch, samp_cov):
        """ Take data that was standardised by transformer, return weighting for every event.
        """
        if self.is_dummy: return None
        
        group_threshold = self.fct_transformation(self.group_value)
        mask = x.detach()[:,self.group_stain_idx] > group_threshold[self.group_arange,self.group_stain_idx][None]
        w = torch.clamp(mask.to(torch.float) * self.group_weight[None], min=1).max(dim=-1).values
        
        w = w*w.shape[0]/w.sum()
        
        return w
    
    def export(self):
        """ Export self to dictionary.
        """
        return {"groups":self.groups}
    
    @classmethod
    def from_exported(cls, exported, transformer):
        """ Restore from exported.
        """
        return cls(transformer, groups=exported["groups"])



