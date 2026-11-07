"""Small torch helpers: straight-through clamp, NaN-safe losses and GPU memory diagnostics."""
import torch
import gc

def _ensure_real_nll(nll):
    """Clip a negative log-likelihood to [0, 1e4] and set NaN entries to 0."""
    return torch.nan_to_num(torch.clamp(nll, min=0, max=1e4), nan=0., posinf=0., neginf=0.)

def _check_weight_sums(w, name="weights"):
    """Raise ValueError unless the event weights `w` sum to a positive, finite value over the last axis.

    Normalising by such a sum would otherwise give a silent NaN/inf loss.
    """
    s = w.sum(dim=-1)
    bad = ~(torch.isfinite(s) & (s>0))
    if bad.any():
        raise ValueError(f"Event {name} must have a positive, finite sum; got {s[bad].flatten()[:5].tolist()}.")

class DifferentiableClamp(torch.autograd.Function):
    """Clamp with a straight-through gradient (autograd function behind `diffclamp`).

    The forward pass is ``torch.clamp``; the backward pass passes the gradient through unchanged, as if the identity
    had been applied. Taken from
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """
    
    @staticmethod
    def forward(ctx, input, min, max):
        """Return ``input.clamp(min=min, max=max)``."""
        return input.clamp(min=min, max=max)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Pass the gradient through unchanged (no gradient for `min` and `max`)."""
        return grad_output.clone(), None, None

def diffclamp(input, min, max):
    """Clamp the values, but let the gradient pass as if no clamp had been applied (straight-through).

    In the package it limits fitted panel embeddings and scale factors and keeps fitted spectra non-negative. Only the
    returned value is clamped: the underlying parameter is not, and can move beyond the bound during the fit (its
    gradient does not vanish there).

    Parameters
    ----------
    input : torch.Tensor
        Values to clamp.
    min : float or None
        Lower bound (None: no lower bound).
    max : float or None
        Upper bound (None: no upper bound).

    Returns
    -------
    torch.Tensor
        ``input.clamp(min, max)``, with gradient 1 with respect to `input` everywhere.
    """
    return DifferentiableClamp.apply(input, min, max)

def torch_print_memory():
    """Print the allocated, reserved and peak reserved CUDA memory (GB) of device 0."""
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def torch_show_tensors_inmemory():
    """Print type and size of every tensor (or object with a tensor ``.data``) tracked by the garbage collector.

    For debugging memory leaks; objects that raise on inspection are skipped.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
