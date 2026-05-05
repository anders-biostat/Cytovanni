import torch
import numpy as np

def is_LossContainer(x):
    return isinstance(x, LossContainer) or x.__class__.__name__=="LossContainer"

def _add_LossContainers(a, b, detach=True):
    """ Add two LossContainers.
    """
    symdiff = set(a.keys()) ^ set(b.keys())
    if len(symdiff)>0: raise ValueError(f"LossContainer add not set up for missing keys! Symmetric difference is {symdiff}.")
    N = a.N + b.N
    factor_a = a.N/N
    factor_b = b.N/N
    newdict = dict()
    d_fct = (lambda x: x.detach()) if detach else (lambda x: x)
    for key in a.keys():
        newdict[key] = factor_a*d_fct(a[key]) + factor_b*d_fct(b[key])
    return LossContainer(newdict, N)

class LossContainer():
    """ Container for losses to simplify averaging etc.
    """
    def __init__(self, dictionary, N=1, allow_partialdictmult=False):
        self.losses = dictionary
        self.__ensure_tensors()
        self.apply_mean()
        self.N = N
        self.allow_partialdictmult = allow_partialdictmult
        
        self.lbd_make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
        self.lbd_make_mean = lambda x: x if x.dim()==0 else x.mean()
        self.lbd_prepare = lambda x: self.lbd_make_mean(self.lbd_make_tensor(x))
        
        self.class_is_loss_container = True
    
    def keys(self):
        return self.losses.keys()
    
    def copy(self):
        """ Return copy of self.
        """
        cp = LossContainer(self.losses.copy(), self.N, allow_partialdictmult=self.allow_partialdictmult)
        return cp
    
    def detach(self):
        """ Detach all losses.
        """
        for key in self.keys():
            self.losses[key] = self.losses[key].detach()
    
    def __ensure_tensors(self):
        """ Ensure all losses are torch tensors.
        """
        for key in self.keys():
            if type(self.losses[key]) is not torch.Tensor: self.losses[key] = torch.tensor(self.losses[key])
    
    def apply_mean(self):
        """ Make sure mean was applied to all losses.
        """
        for key in self.keys():
            if self.losses[key].dim()>0: self.losses[key] = self.losses[key].mean() 
    
    def to_numberdict(self):
        """ Return self as regular dict with numbers.
        """
        return {key: self[key].detach().cpu().item() for key in self.keys()}
    
    def update(self, dictionary):
        """ Update with key value pairs from dictionary,
            can be both dict or LossContainer.
        """
        if dictionary is not None:
            inters = set(self.keys()).intersection(dictionary.keys())
            if len(inters)>0: raise ValueError(f"Can't update, has intersecting keys {inters}.")
            if type(dictionary) is LossContainer and dictionary.N!=self.N:
                raise ValueError(f"Trying to update LossContainer with loss that has different sample size, sizes {self.N} and {dictionary.N}!")
            for key in dictionary.keys():
                self[key] = dictionary[key]
    
    def to_module_log(self, module, prefix=""):
        for key in self.keys():
            module.log(prefix+key, self[key], batch_size=self.N)
    
    def __multiply_losses(self, factor):
        """ Global factor for all losses.
        """
        for key in self.keys():
            self.losses[key] = self.losses[key]*factor
    
    def __multiply_losses_fromdict(self, factordict):
        """ Global factor for all losses.
        """
        if not self.allow_partialdictmult and set(self.keys()).intersection(set(factordict.keys())) != set(self.keys()):
            raise ValueError("Trying to multiply losses with factors from dict that is missing some keys!")
        
        for key in [key for key in self.keys() if key in factordict.keys()]:
            self.losses[key] = self.losses[key]*factordict[key]
    
    def __getitem__(self, key):
        """ Dict access.
            If key is list of keys, return sum of the values.
        """
        if type(key) is list:
            return sum([self.losses[k] for k in key])
        return self.losses[key]
    
    def __setitem__(self, key, value):
        self.losses[key] = self.lbd_prepare(value)
    
    def __repr__(self):
        return f"LossContainer with {len(self.losses)} losses."
    
    def __radd__(self, x):  return self.__add__(x)
    def __add__(self, x):
        """ Detaches all losses when adding containers!
        """
        #print(type(x))
        if x is None:                  return self
        elif is_LossContainer(x): return _add_LossContainers(self, x)
        else:                          raise NotImplementedError(f"Adding {type(self)} and type {type(x)} is not supported!")
    
    def __rmul__(self, x): return self.__mul__(x)
    def __mul__(self, x):
        """ Why do I copy this?
        """
        if type(x) is dict:
            res = self.copy()
            res.__multiply_losses_fromdict(x)
            return res
        else:
            try:
                factor = float(x)
                res = self.copy()
                res.__multiply_losses(factor)
                return res
            except TypeError:
                raise NotImplementedError()

