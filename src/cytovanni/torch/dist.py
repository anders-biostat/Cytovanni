import math
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Transform, TransformedDistribution, MultivariateNormal, Normal
from torch.distributions.utils import _sum_rightmost, broadcast_all
from torch.distributions.utils import _standard_normal, lazy_property
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv, _precision_to_scale_tril

class CustomMultivariateNormal(Distribution):
    """ Basically simple version of MultivariateNormal, but takes precision and its determinant
        in cases where it is more stable to calculate them beforehand instead of using the decomposition as MultivariateNormal does.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "precision_matrix": constraints.real,
        "precision_logdet": constraints.real,
    }
    support = constraints.real_vector

    def __init__(
        self,
        loc,
        precision_matrix,
        precision_logdet,
        validate_args=None,
    ):

        batch_shape = torch.broadcast_shapes( precision_matrix.shape[:-2], loc.shape[:-1] )
        event_shape = loc.shape[-1:]
        
        self.loc = loc.expand(batch_shape + (-1,))
        self.precision_matrix = precision_matrix
        self.precision_logdet = precision_logdet
        
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    
    @lazy_property
    def _unbroadcasted_scale_tril(self):
         return _precision_to_scale_tril(self.precision_matrix)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = dist = torch.einsum("...i, ...ij, ...j -> ...", diff, self.precision_matrix, diff)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) +0.5 * self.precision_logdet

    def rsample(self, sample_shape=torch.Size()):
        eps = _standard_normal(self.loc.shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

class CustomTransformedDistribution(TransformedDistribution):
    """ Implements .log_prob_onbase that ignores the Jacobian of the transforms that is used in .log_prob,
        also allows passing through additional arguments to log_prob.
    """

    def log_prob_onbase(self, value, *args, **kwargs):
        """
            Scores the sample by inverting the transform(s) and computing the score
            using the score of the base distribution.
            Ignores Jacobians of transforms.
        """
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        y = value
        for transform in reversed(self.transforms):
            y = transform.inv(y)

        log_prob = _sum_rightmost( self.base_dist.log_prob(y, *args, **kwargs), event_dim - len(self.base_dist.event_shape) )
        return log_prob

    def log_prob(self, value, *args, **kwargs):
        """
            Scores the sample by inverting the transform(s) and computing the score
            using the score of the base distribution and the log abs det jacobian.
        """
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x

        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(y, *args, **kwargs), event_dim - len(self.base_dist.event_shape)
        )
        return log_prob

class HyperbolicSineTransform(Transform):
    r"""
        Transform via the mapping :math:`y = \sinh(x)`.
        
        https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.Transform
        
        d sinh(x) /dx = cosh(x)>0
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, HyperbolicSineTransform)

    def _call(self, x):
        return x.sinh()

    def _inverse(self, y):
        return y.asinh()

    def log_abs_det_jacobian(self, x, y):
        """ Computes the log det jacobian, log |dy/dx|, given input and output.
        """
        return x.cosh().log()

class ASinhMultivariateNormal(CustomTransformedDistribution):
    r"""
        Inverse hyperbolic sine multivariate normal distribution, modelled after LogNormal
        https://github.com/pytorch/pytorch/blob/main/torch/distributions/log_normal.py
        
        Optionally uses CustomMultivariateNormal with precomputed precision logdet
    """
    arg_constraints = {
        # MultivariateNormal already tests all the args
        # CustomMultivariateNormal doesn't but that is intentional, make sure they are fine beforehand
    }
    
    support = constraints.real
    has_rsample = True

    def __init__(self,
                 loc=None, loc_raw=None,
                 covariance_matrix=None, precision_matrix=None, scale_tril=None, precision_logdet=None,
                 validate_args=None):
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError("Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.")
        if (loc is not None) + (loc_raw is not None) != 1:
            raise ValueError("Exactly one of loc or loc_raw may be specified.")
        
        use_custom = (precision_logdet is not None) and (precision_matrix is not None)
        
        transform = HyperbolicSineTransform()
        if use_custom:
            base_dist = CustomMultivariateNormal(transform._inverse(loc_raw) if loc is None else loc,
                                                 precision_matrix=precision_matrix,
                                                 precision_logdet=precision_logdet,
                                                 validate_args=validate_args)
        else:
            base_dist = MultivariateNormal(transform._inverse(loc_raw) if loc is None else loc,
                                           covariance_matrix=covariance_matrix,
                                           precision_matrix=precision_matrix,
                                           scale_tril=scale_tril,
                                           validate_args=validate_args)
        super().__init__(base_dist, transform, validate_args=validate_args)
    
    def __repr__(self):
        return "ASinh transformed "+self.base_dist.__repr__()

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def covariance_matrix(self):
        return self.base_dist.covariance_matrix

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def median(self):
        return self.base_dist.loc.sinh()

    @property
    def variance(self):
        raise NotImplementedError
