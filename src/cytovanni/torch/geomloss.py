import torch
from torch.nn import Module, Parameter
import numpy as np

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
            # not sure why this needs a kernel and sinkhorn doesn't, just use Gaussian
            self.loss = geomloss.SamplesLoss(loss="hausdorff", p=2, blur=blur, reach=reach, scaling=scaling, backend="tensorized", kernel=geomloss.kernel_samples.gaussian_kernel)
        else:
            raise NotImplemented(f"GeomSampleLoss {mode} is not available!")
        
        

    def forward(self, X, Y, wx=None, wy=None):
        """ SampleLoss.forward has weird syntax.

            Weights should be normalized to one, but this is not enforced here!
            Make sure you check this beforehand wherever the weights come form.
        """
        if wx is None or wy is None:
            return self.loss(X, Y)
        else:
            return self.loss(wx, X, wy, Y)
                
            