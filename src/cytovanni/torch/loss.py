"""Named scalar losses with event-count weighted averaging, for logging fits."""
import torch
import numpy as np

def is_LossContainer(x):
    """Return True if `x` is a `LossContainer` (also matched by class name only)."""
    return isinstance(x, LossContainer) or x.__class__.__name__=="LossContainer"

def _add_LossContainers(a, b, detach=True):
    """Average two LossContainers with the same keys, weighted by their `N`; detached unless `detach` is False."""
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
    """Dict of named scalar losses with an event count, for averaging and logging.

    Every loss is stored as a 0-dim tensor (non-tensors are converted, tensors with more dimensions are replaced by
    their mean). Adding two containers gives the `N`-weighted average of each loss (detached), so per-batch losses
    can be accumulated over an epoch, e.g. ``total = sum(containers, None)`` (the built-in ``sum`` with its default
    start value 0 raises NotImplementedError). Multiplying by a number or a dict of factors returns a scaled copy.

    Parameters
    ----------
    dictionary : dict of str to torch.Tensor or float
        Losses by name. The dict is used directly (not copied) and converted in place.
    N : int, default 1
        Number of events (or samples) the losses were computed on; the weight when adding containers.
    allow_partialdictmult : bool, default False
        If True, multiplying with a dict of factors scales only the keys present in the dict; if False, the dict
        must contain every key.

    Examples
    --------
    >>> lc = LossContainer({"ddist": torch.tensor([1., 3.])}, N=2)
    >>> (lc + LossContainer({"ddist": 4.}, N=6))["ddist"]
    tensor(3.5000)
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
        """Return the loss names (dict keys view)."""
        return self.losses.keys()
    
    def copy(self):
        """Return a new container with a shallow copy of the loss dict and the same `N`."""
        cp = LossContainer(self.losses.copy(), self.N, allow_partialdictmult=self.allow_partialdictmult)
        return cp
    
    def detach(self):
        """Detach all losses from the autograd graph, in place."""
        for key in self.keys():
            self.losses[key] = self.losses[key].detach()
    
    def __ensure_tensors(self):
        """Convert all losses to torch tensors, in place."""
        for key in self.keys():
            if type(self.losses[key]) is not torch.Tensor: self.losses[key] = torch.tensor(self.losses[key])
    
    def apply_mean(self):
        """Replace every loss with more than 0 dimensions by its mean, in place."""
        for key in self.keys():
            if self.losses[key].dim()>0: self.losses[key] = self.losses[key].mean() 
    
    def to_numberdict(self):
        """Return the losses as a plain dict of Python floats (detached, on the CPU)."""
        return {key: self[key].detach().cpu().item() for key in self.keys()}
    
    def update(self, dictionary):
        """Add new losses from a dict or LossContainer, in place.

        Parameters
        ----------
        dictionary : dict or LossContainer or None
            Losses to add (None: nothing happens). Values are converted as in the constructor.

        Raises
        ------
        ValueError
            If a key is already present, or `dictionary` is a LossContainer with a different `N`.
        """
        if dictionary is not None:
            inters = set(self.keys()).intersection(dictionary.keys())
            if len(inters)>0: raise ValueError(f"Can't update, has intersecting keys {inters}.")
            if type(dictionary) is LossContainer and dictionary.N!=self.N:
                raise ValueError(f"Trying to update LossContainer with loss that has different sample size, sizes {self.N} and {dictionary.N}!")
            for key in dictionary.keys():
                self[key] = dictionary[key]
    
    def to_module_log(self, module, prefix=""):
        """Log every loss with ``module.log(prefix + key, value, batch_size=N)`` (PyTorch Lightning style).

        Parameters
        ----------
        module : object
            Object with a Lightning-style ``log`` method.
        prefix : str, default ""
            Prefix of the logged names, e.g. ``"train_"``.
        """
        for key in self.keys():
            module.log(prefix+key, self[key], batch_size=self.N)
    
    def __multiply_losses(self, factor):
        """Multiply all losses by one factor, in place."""
        for key in self.keys():
            self.losses[key] = self.losses[key]*factor
    
    def __multiply_losses_fromdict(self, factordict):
        """Multiply each loss by its factor from `factordict`, in place (all keys required unless allowed)."""
        if not self.allow_partialdictmult and set(self.keys()).intersection(set(factordict.keys())) != set(self.keys()):
            raise ValueError("Trying to multiply losses with factors from dict that is missing some keys!")
        
        for key in [key for key in self.keys() if key in factordict.keys()]:
            self.losses[key] = self.losses[key]*factordict[key]
    
    def __getitem__(self, key):
        """Return the loss `key`, or the sum of the losses if `key` is a list of names."""
        if type(key) is list:
            return sum([self.losses[k] for k in key])
        return self.losses[key]
    
    def __setitem__(self, key, value):
        """Set loss `key` (converted to a tensor; mean taken if it has more than 0 dimensions)."""
        self.losses[key] = self.lbd_prepare(value)
    
    def __repr__(self):
        return f"LossContainer with {len(self.losses)} losses."
    
    def __radd__(self, x):  return self.__add__(x)
    def __add__(self, x):
        """Return the `N`-weighted average with another LossContainer (detached); ``self + None`` returns self.

        Raises
        ------
        ValueError
            If the two containers have different keys.
        NotImplementedError
            If `x` is neither None nor a LossContainer.
        """
        #print(type(x))
        if x is None:                  return self
        elif is_LossContainer(x): return _add_LossContainers(self, x)
        else:                          raise NotImplementedError(f"Adding {type(self)} and type {type(x)} is not supported!")
    
    def __rmul__(self, x): return self.__mul__(x)
    def __mul__(self, x):
        """Return a scaled copy: all losses times a number, or each loss times its factor from a dict.

        The container itself is not changed.

        Raises
        ------
        ValueError
            If `x` is a dict that misses keys of the container and `allow_partialdictmult` is False.
        NotImplementedError
            If `x` is not a dict and ``float(x)`` raises TypeError (e.g. None or a list).
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

