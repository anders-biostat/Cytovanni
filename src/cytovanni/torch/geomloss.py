import torch
from torch.nn import Module, Parameter
import numpy as np

from .utils import _check_weight_sums

class GeomSampleLoss(Module):
    """ Wrapper for SampleLoss from geomloss.
        Always use tensorized backend, we use low enough event counts that it will anywa get chosen.
        
        Probably better to copy/reimplement the parts I need?
        geomloss depends on keops, which is annoying to get to work, but I don't need that part.
    """
    def __init__(self, mode, blur=.5, reach=1., scaling=.5):
        """ 'hausdorff' seems faster than 'sinkhorn', but needs more iterations to converge?
            
            :param mode: str. Which version of the loss to use; 'sinkhorn' and 'hausdorff' are available.
            
            :param blur: float. For 'sinkhorn'/'hausdorff', interpolates between Wasserstein/ICP (blur=0) and kernel (blur=+inf). Default .05 is meant for data in unit square/cube, .5 seems to work for the 23 color panel at cofactor 5000, not sure how this scales. Probably would be useful to implement something that sets this based on the data like the MMD bandwidth; this is anyway something like a kernel width: http://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html
            
            :param reach: float. Lower achieves better match for unbalanced data, but also converges slower; 1. seems like a reasonable tradeoff, 'hausdorff' needs a bit more, maybe 5.
            
            :param scaling: float. Trade-off between speed (scaling<.4) and accuracy (scaling>.9).
        """
        super().__init__()
        self.mode = mode
        
        import geomloss
        
        if mode=="sinkhorn":
            self.loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=blur, reach=reach, scaling=scaling, backend="tensorized")
        elif mode=="hausdorff":
            try: # submodule, not loaded by `import geomloss`; moved to geomloss._legacy in geomloss 0.3
                from geomloss.kernel_samples import gaussian_kernel
            except ImportError:
                from geomloss._legacy.kernel_samples import gaussian_kernel
            # not sure why this needs a kernel and sinkhorn doesn't, just use Gaussian
            self.loss = geomloss.SamplesLoss(loss="hausdorff", p=2, blur=blur, reach=reach, scaling=scaling, backend="tensorized", kernel=gaussian_kernel)
        else:
            raise NotImplementedError(f"GeomSampleLoss {mode} is not available!")
        
        

    def forward(self, X, Y, wx=None, wy=None):
        """ Loss between the point clouds X (N, D) and Y (M, D) (or batched, (B, N, D)), optionally with per-event weights.

            geomloss with finite `reach` is unbalanced, so the total mass matters. Each weight vector is therefore
            normalised to sum 1 here (e.g. RareMarkerWeights sums to N, which is right for MMD but not for this loss).
            A missing side (None) gets uniform weights 1/N; with both None this is the unweighted loss.

            :param wx: tensor (N,) (batched: (B, N)) or None. Non-negative weights of X, with a positive, finite sum
                (ValueError otherwise).
            :param wy: tensor (M,) (batched: (B, M)) or None. Non-negative weights of Y, as wx.
        """
        if wx is None and wy is None:
            return self.loss(X, Y)
        wx = self._normalised_weights(wx, X)
        wy = self._normalised_weights(wy, Y)
        return self.loss(wx, X, wy, Y)

    @staticmethod
    def _normalised_weights(w, X):
        """ Weights normalised to sum 1 over the events (last axis; also for batched input X (B, N, D), w (B, N)).
            Raises ValueError if a weight sum is not positive and finite.
        """
        if w is None:
            return torch.full(X.shape[:-1], 1. / X.shape[-2], dtype=X.dtype, device=X.device)
        w = w.to(dtype=X.dtype)
        _check_weight_sums(w)
        return w / w.sum(dim=-1, keepdim=True)
