"""Maximum mean discrepancy (MMD) loss with a multi-scale Gaussian (RBF) kernel and optional event weights.

Adapted from https://github.com/yiftachbeer/mmd_loss_pytorch. The bandwidth is a squared distance: the kernel is a
sum of ``exp(-|x - y|**2 / (bandwidth * m))`` over several multipliers m.
"""
import torch
from torch.nn import Module, Parameter
import numpy as np

from .utils import _check_weight_sums

def _bandwidth_floor(bandwidth):
    """Clamp an estimated bandwidth to a small positive value, so identical points give a finite kernel, not 0/0.

    The floor is the square root of the smallest normal number of the bandwidth's dtype, but at least that of float32
    (about 1e-19), because the kernel multiplies the bandwidth with float32 multipliers (and estimated bandwidths are
    reused as fixed ones). Ordinary bandwidths are far above it.
    """
    tiny = max(torch.finfo(bandwidth.dtype).tiny, torch.finfo(torch.float32).tiny)
    return bandwidth.clamp_min(tiny ** .5)

def subsamplemean_MMD(x, y, mmd=None, device="cpu", M=20, N=2000):
    """Estimate the MMD between two large samples as the mean over repeated random subsamples.

    The MMD needs memory and time quadratic in the number of events; this draws `N` events from each sample (without
    replacement, all events if fewer), computes the loss, and repeats `M` times.

    Parameters
    ----------
    x : numpy.ndarray of shape (events, D)
        First sample, on the scale the loss should compare (e.g. arcsinh-transformed).
    y : numpy.ndarray of shape (events, D)
        Second sample, same D.
    mmd : MMDLoss, optional
        Loss module to use; it is moved to `device`. Default: ``MMDLoss()``, which estimates the bandwidth anew for
        every subsample pair.
    device : str, default "cpu"
        Torch device for the computation.
    M : int, default 20
        Number of repetitions.
    N : int, default 2000
        Events per subsample.

    Returns
    -------
    mean : float
        Mean MMD over the repetitions.
    sem : float
        Standard error of that mean (standard deviation over the repetitions / sqrt(M)).
    """
    if mmd is None: mmd = MMDLoss()
    mmd.to(device)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def get_single():
        return mmd(x[torch.randperm(x.shape[0])[:N]].to(device),
                   y[torch.randperm(y.shape[0])[:N]].to(device)).item()
    dist = [get_single() for i in range(M)]
    return np.mean(dist), np.std(dist)/np.sqrt(M)

def batched_cdist(x, y):
    """Euclidean distances between all pairs of points, shapes (..., N, D) and (..., M, D) -> (..., N, M)."""
    return torch.sqrt(((x[..., :, None, :] - y[..., None, :, :]) ** 2).sum(-1))

# Taken from https://github.com/yiftachbeer/mmd_loss_pytorch
class RBF(Module):
    """Sum of Gaussian kernels with several bandwidths, evaluated on all pairs of one sample.

    ``K(x, y) = sum_k exp(-|x - y|**2 / (bandwidth * m_k))`` with multipliers
    ``m_k = mul_factor ** (k - n_kernels // 2)``, k = 0 .. n_kernels - 1 (by default 1/4, 1/2, 1, 2, 4).

    Parameters
    ----------
    n_kernels : int, default 5
        Number of kernels.
    mul_factor : float, default 2.0
        Ratio between successive bandwidths.
    bandwidth : float or torch.Tensor, optional
        Fixed bandwidth (a squared distance, in squared data units). Default: estimated on every call as the mean
        squared distance between distinct points of the input (see `estimate_bandwidth`).
    """
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = Parameter(mul_factor ** (torch.arange(n_kernels) - n_kernels // 2), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """Return the fixed bandwidth, or the mean off-diagonal entry of `L2_distances` (N, N) (floored)."""
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return _bandwidth_floor(L2_distances.data.mean() / (1 - 1/n_samples)) # mean across non-diagonal
        return self.bandwidth
    
    @classmethod
    def estimate_bandwidth(cls, X):
        """Estimate a bandwidth from one sample, to fix it for later comparisons.

        A fixed bandwidth makes losses comparable across different sample pairs (the default re-estimates it on
        every call).

        Parameters
        ----------
        X : torch.Tensor of shape (N, D)
            Sample, on the scale of the later comparisons.

        Returns
        -------
        torch.Tensor
            0-dim tensor: the mean squared Euclidean distance between distinct points of `X`, floored at a small
            positive value.
        """
        L2_distances = torch.cdist(X, X) ** 2
        n_samples = L2_distances.shape[0]
        return _bandwidth_floor(L2_distances.data.mean() / (1 - 1/n_samples)) # mean across non-diagonal

    def forward(self, X, *args, **kwargs):
        """Return the kernel matrix of shape (N, N) for the points `X` of shape (N, D); extra arguments are ignored."""
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(Module):
    """Maximum mean discrepancy between two samples with the multi-scale `RBF` kernel.

    Adapted from https://github.com/yiftachbeer/mmd_loss_pytorch. The estimate is the biased (V-statistic) one:
    means over all kernel entries, including the diagonal. Optional event weights follow
    https://proceedings.mlr.press/v161/bellot21b.html, here for both samples.

    Parameters
    ----------
    **kwargs_kernel
        Passed to `RBF`: `n_kernels`, `mul_factor`, `bandwidth`. Without a fixed `bandwidth`, it is estimated on
        every call from the pooled sample ``vstack([X, Y])`` (floored at a small positive value).

    See Also
    --------
    GeomSampleLoss : Sinkhorn divergence, the default distribution loss of the reference model.
    """
    def __init__(self, **kwargs_kernel):
        super().__init__()
        self.kernel = RBF(**kwargs_kernel)
    
    @classmethod
    def estimate_bandwidth(cls, X):
        """Estimate a bandwidth from the sample `X` (N, D), see `RBF.estimate_bandwidth`; returns a 0-dim tensor."""
        return RBF.estimate_bandwidth(X)

    def forward(self, X, Y, wx=None, wy=None):
        """Compute the (weighted) MMD between two samples.

        Parameters
        ----------
        X : torch.Tensor of shape (N, D)
            First sample.
        Y : torch.Tensor of shape (M, D)
            Second sample, same D.
        wx : torch.Tensor of shape (N,), optional
            Non-negative event weights of `X`, normalised here to mean 1. Default: all 1.
        wy : torch.Tensor of shape (M,), optional
            Non-negative event weights of `Y`, as `wx`.

        Returns
        -------
        torch.Tensor
            0-dim tensor, ``mean(wx K_XX wx) - 2 mean(wx K_XY wy) + mean(wy K_YY wy)``.

        Raises
        ------
        ValueError
            If a given weight vector does not have a positive, finite sum.
        """
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size]
        XY = K[:X_size, X_size:]
        YY = K[X_size:, X_size:]
        if wx is None and wy is None:
            return XX.mean() - 2 * XY.mean() + YY.mean()
        else:
            for w, name in ((wx, "weights wx"), (wy, "weights wy")):
                if w is not None: _check_weight_sums(w, name)
            wx = torch.ones_like(X[:,0]) if wx is None else wx*(wx.shape[0]/wx.sum())
            wy = torch.ones_like(Y[:,0]) if wy is None else wy*(wy.shape[0]/wy.sum())
            return (wx[:,None] * XX * wx[None,:]).mean() - 2 * (wx[:,None] * XY * wy[None,:]).mean() + (wy[:,None] * YY * wy[None,:]).mean()



# Taken from https://github.com/yiftachbeer/mmd_loss_pytorch, with modifications for batched processing
class RBF_batched(Module):
    """Batched version of `RBF` for inputs of shape (..., N, D); the estimated bandwidth is per batch, not floored.

    Parameters
    ----------
    n_kernels : int, default 5
        Number of kernels.
    mul_factor : float, default 2.0
        Ratio between successive bandwidths.
    bandwidth : float or torch.Tensor, optional
        Fixed bandwidth (squared distance). Default: estimated on every call, per batch.
    """
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = Parameter(mul_factor ** (torch.arange(n_kernels) - n_kernels // 2), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """Return the fixed bandwidth, or the mean off-diagonal entry of `L2_distances` (..., N, N) per batch."""
        if self.bandwidth is None:
            n_samples = L2_distances.shape[-2]
            return L2_distances.data.mean(-1, keepdim=True).mean(-2, keepdim=True) / (1 - 1/n_samples) # mean across non-diagonal
        bandwidth = torch.as_tensor(self.bandwidth, dtype=L2_distances.dtype, device=L2_distances.device)
        if bandwidth.ndim == 0:
            bandwidth = bandwidth.reshape(1, 1)
        return bandwidth
    
    @classmethod
    def estimate_bandwidth(cls, X):
        """Estimate the bandwidth per batch from `X` (..., N, D): mean squared distance between distinct points.

        Returns a tensor of shape (..., 1, 1) (not floored, unlike `RBF.estimate_bandwidth`).
        """
        L2_distances = batched_cdist(X, X) ** 2
        n_samples = L2_distances.shape[-2]
        return L2_distances.data.mean(-1, keepdim=True).mean(-2, keepdim=True) / (1 - 1/n_samples) # mean across non-diagonal

    def forward(self, X, *args, **kwargs):
        """Return the kernel matrices of shape (..., N, N) for `X` of shape (..., N, D); extra arguments are ignored."""
        L2_distances = batched_cdist(X, X) ** 2
        nl2 = -L2_distances[..., None, :, :]
        bw = self.get_bandwidth(L2_distances)[..., None, :, :]
        bwm = self.bandwidth_multipliers[:, None, None][(None,) * (nl2.ndim-3) + (...,)]
        return torch.exp(nl2 / (bw * bwm)).sum(dim=-3)

class MMDLoss_batched(Module):
    """Unfinished batched version of `MMDLoss`: the constructor raises NotImplementedError.

    Parameters
    ----------
    **kwargs_kernel
        Would be passed to `RBF_batched`.

    Raises
    ------
    NotImplementedError
        Always, on construction.
    """
    def __init__(self, **kwargs_kernel):
        super().__init__()
        raise NotImplementedError("This is only partially implemented!")
        self.kernel = RBF_batched(**kwargs_kernel)
    
    @classmethod
    def estimate_bandwidth(cls, X):
        """Estimate a bandwidth with the unbatched `RBF.estimate_bandwidth`."""
        return RBF.estimate_bandwidth(X)

    def forward(self, X, Y, wx=None, wy=None):
        """Unbatched MMD computation copied from `MMDLoss.forward` (not reachable, see the class)."""
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size]
        XY = K[:X_size, X_size:]
        YY = K[X_size:, X_size:]
        if wx is None and wy is None:
            return XX.mean() - 2 * XY.mean() + YY.mean()
        else:
            wx = torch.ones_like(X[:,0]) if wx is None else wx*(wx.shape[0]/wx.sum())
            wy = torch.ones_like(Y[:,0]) if wy is None else wy*(wy.shape[0]/wy.sum())
            return (wx[:,None] * XX * wx[None,:]).mean() - 2 * (wx[:,None] * XY * wy[None,:]).mean() + (wy[:,None] * YY * wy[None,:]).mean()
                
            