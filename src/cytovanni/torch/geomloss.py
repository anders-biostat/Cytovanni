"""Optimal-transport losses between point clouds (wrapper around ``geomloss.SamplesLoss``)."""
import torch
from torch.nn import Module, Parameter
import numpy as np

from .utils import _check_weight_sums

class GeomSampleLoss(Module):
    """Sinkhorn or Hausdorff divergence between two point clouds, as a torch module.

    Wraps ``geomloss.SamplesLoss`` with squared Euclidean cost (p=2) and always uses its tensorized backend (dense
    cost matrices, memory grows with N * M; geomloss would choose this backend anyway for the subsample sizes used in
    the package). Used as the distribution loss of the reference-model and factor fits, on arcsinh-transformed data.

    Parameters
    ----------
    mode : {'sinkhorn', 'hausdorff'}
        Loss to use: the debiased Sinkhorn divergence, or geomloss's Hausdorff divergence with a Gaussian kernel.
    blur : float, default 0.5
        Length scale below which differences between the clouds hardly matter, in the units of the data (here
        arcsinh units); the entropic regularisation is ``eps = blur**2``. It interpolates between exact optimal
        transport (blur -> 0) and a kernel distance (blur -> inf), see
        http://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html.
        geomloss's own default, 0.05, is meant for data in the unit cube; 0.5 was chosen for a 23-colour panel at
        arcsinh cofactor 5000 (noise within about +-0.4, highest expression about 4) and may need a different value
        for data on another scale.
    reach : float or None, default 1.0
        Distance above which differences are treated as differences in composition rather than shifts (unbalanced
        transport: mass can be created or destroyed at a cost, marginal penalty ``rho = reach**2``); None gives
        balanced transport. Should be larger than `blur`. In the author's experience, a lower reach matches
        unbalanced data better but converges more slowly; 1 is a compromise for 'sinkhorn', 'hausdorff' may need
        more (about 5).
    scaling : float, default 0.5
        Ratio between successive blur values of geomloss's eps-scaling: a trade-off between speed (below about 0.4)
        and accuracy (above about 0.9).

    Raises
    ------
    NotImplementedError
        If `mode` is not 'sinkhorn' or 'hausdorff'.
    """
    def __init__(self, mode, blur=.5, reach=1., scaling=.5):
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
        """Compute the loss between two point clouds, optionally with per-event weights.

        With a finite `reach` the transport is unbalanced, so the total mass matters: each weight vector is
        normalised to sum 1 here (event weights such as those of `RareMarkerWeights` sum to the number of events,
        which suits the MMD loss but not this one). A missing side (None) gets uniform weights 1/N; with both None
        this is the unweighted loss.

        Parameters
        ----------
        X : torch.Tensor of shape (N, D) or (B, N, D)
            First point cloud (events, dimensions), optionally with a leading batch axis.
        Y : torch.Tensor of shape (M, D) or (B, M, D)
            Second point cloud, same D (and B).
        wx : torch.Tensor of shape (N,) or (B, N), optional
            Non-negative weights of the events of `X`.
        wy : torch.Tensor of shape (M,) or (B, M), optional
            Non-negative weights of the events of `Y`.

        Returns
        -------
        torch.Tensor
            Scalar loss, or shape (B,) for batched input.

        Raises
        ------
        ValueError
            If a given weight vector does not have a positive, finite sum.
        """
        if wx is None and wy is None:
            return self.loss(X, Y)
        wx = self._normalised_weights(wx, X)
        wy = self._normalised_weights(wy, Y)
        return self.loss(wx, X, wy, Y)

    @staticmethod
    def _normalised_weights(w, X):
        """Return `w` normalised to sum 1 over the last axis (uniform 1/N if None); ValueError for a bad sum."""
        if w is None:
            return torch.full(X.shape[:-1], 1. / X.shape[-2], dtype=X.dtype, device=X.device)
        w = w.to(dtype=X.dtype)
        _check_weight_sums(w)
        return w / w.sum(dim=-1, keepdim=True)
