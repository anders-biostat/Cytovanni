import torch
import gc

def _ensure_real_nll(nll):
    """ Ensure that nll is real, set to zero otherwise, clip to [0, 1e4].
    """
    return torch.nan_to_num(torch.clamp(nll, min=0, max=1e4), nan=0., posinf=0., neginf=0.)

def _check_weight_sums(w, name="weights"):
    """ Raise ValueError unless the event weights w sum to a positive, finite value (over the last axis).
        Normalising by such a sum would otherwise give a silent NaN/inf loss.
    """
    s = w.sum(dim=-1)
    bad = ~(torch.isfinite(s) & (s>0))
    if bad.any():
        raise ValueError(f"Event {name} must have a positive, finite sum; got {s[bad].flatten()[:5].tolist()}.")

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    
    Taken from https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    """
    
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def diffclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

def torch_print_memory():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def torch_show_tensors_inmemory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
