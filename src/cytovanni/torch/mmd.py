import torch
from torch.nn import Module, Parameter
import numpy as np

def subsamplemean_MMD(x, y, mmd=None, device="cpu", M=20, N=2000):
    """ Subsample both x and y to N, calculate MMD loss M times, average.
        To avoid the quadratic scaling of the MMD loss.
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
    return torch.sqrt(((x[..., :, None, :] - y[..., None, :, :]) ** 2).sum(-1))

# Taken from https://github.com/yiftachbeer/mmd_loss_pytorch
class RBF(Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = Parameter(mul_factor ** (torch.arange(n_kernels) - n_kernels // 2), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.mean() / (1 - 1/n_samples) # mean across non-diagonal
        return self.bandwidth
    
    @classmethod
    def estimate_bandwidth(cls, X):
        """ Estimate bandwidth here to make it consistent across different samples.
            This approach to estimating the bandwidth seems pretty optimal for our use case.
        """
        L2_distances = torch.cdist(X, X) ** 2
        n_samples = L2_distances.shape[0]
        return L2_distances.data.mean() / (1 - 1/n_samples) # mean across non-diagonal

    def forward(self, X, *args, **kwargs):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(Module):
    """ Maximum Mean Discreptancy loss, taken from https://github.com/yiftachbeer/mmd_loss_pytorch with RBF.
        Technically .mean yields the biased estimator (1/(n-1) would be unbiased), irrelevant
        
        Implement weighting as in https://proceedings.mlr.press/v161/bellot21b.html but with weights for both x and y.
    """
    def __init__(self, **kwargs_kernel):
        super().__init__()
        self.kernel = RBF(**kwargs_kernel)
    
    @classmethod
    def estimate_bandwidth(cls, X):
        return RBF.estimate_bandwidth(X)

    def forward(self, X, Y, wx=None, wy=None):
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



# Taken from https://github.com/yiftachbeer/mmd_loss_pytorch, with modifications for batched processing
class RBF_batched(Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = Parameter(mul_factor ** (torch.arange(n_kernels) - n_kernels // 2), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[-2]
            return L2_distances.data.mean(-1, keepdim=True).mean(-2, keepdim=True) / (1 - 1/n_samples) # mean across non-diagonal
        return self.bandwidth
    
    @classmethod
    def estimate_bandwidth(cls, X):
        """ Estimate bandwidth here to make it consistent across different samples.
            This approach to estimating the bandwidth seems pretty optimal for our use case.
        """
        L2_distances = batched_cdist(X, X) ** 2
        n_samples = L2_distances.shape[-2]
        return L2_distances.data.mean(-1, keepdim=True).mean(-2, keepdim=True) / (1 - 1/n_samples) # mean across non-diagonal

    def forward(self, X, *args, **kwargs):
        L2_distances = batched_cdist(X, X) ** 2
        nl2 = -L2_distances[..., None, :, :]
        bw = self.get_bandwidth(L2_distances)[..., None, :, :]
        bwm = self.bandwidth_multipliers[:, None, None][(None,) * (nl2.ndim-3) + (...,)]
        return torch.exp(nl2 / (bw * bwm)).sum(dim=-3)

class MMDLoss_batched(Module):
    """ Maximum Mean Discreptancy loss, taken from https://github.com/yiftachbeer/mmd_loss_pytorch with RBF.
        Technically .mean yields the biased estimator (1/(n-1) would be unbiased), irrelevant
        
        Implement weighting as in https://proceedings.mlr.press/v161/bellot21b.html but with weights for both x and y.
    """
    def __init__(self, **kwargs_kernel):
        super().__init__()
        raise NotImplementedError("This is only partially implemented!")
        self.kernel = RBF_batched(**kwargs_kernel)
    
    @classmethod
    def estimate_bandwidth(cls, X):
        return RBF.estimate_bandwidth(X)

    def forward(self, X, Y, wx=None, wy=None):
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
                
            