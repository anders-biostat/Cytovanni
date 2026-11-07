"""Probability distributions for the rainbow bead model: a multivariate normal on the arcsinh scale.

`ASinhMultivariateNormal` is the distribution of ``sinh(z)`` with ``z`` multivariate normal, i.e. the data
``y`` is modelled as normal after ``arcsinh(y)``. The rainbow model applies it to intensities divided by the arcsinh
cofactor (plus a fitted shift), so ``arcsinh(y)`` is essentially the usual ``arcsinh(x / cofactor)``.
"""
import math
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Transform, TransformedDistribution, MultivariateNormal, Normal
from torch.distributions.utils import _sum_rightmost, broadcast_all
from torch.distributions.utils import _standard_normal, lazy_property
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv, _precision_to_scale_tril

class CustomMultivariateNormal(Distribution):
    """Multivariate normal given by its precision matrix and the log-determinant of the precision.

    A simplified ``torch.distributions.MultivariateNormal`` for cases where the precision and its log-determinant are
    computed more stably beforehand (e.g. from a factor ``A`` with precision ``A A^T``, see
    `PrecisionExtendableParameter`) than from the Cholesky decomposition MultivariateNormal uses. The arguments are not
    validated (`arg_constraints` are only nominal): make sure they are consistent.

    Parameters
    ----------
    loc : torch.Tensor of shape (..., D)
        Mean.
    precision_matrix : torch.Tensor of shape (..., D, D)
        Precision (inverse covariance) matrix; leading axes broadcast with those of `loc`.
    precision_logdet : torch.Tensor of shape (...)
        ``log det(precision_matrix)``.
    validate_args : bool, optional
        Passed to ``torch.distributions.Distribution`` (only checks the samples passed to `log_prob`).
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
        """Return the log density at `value` (..., D).

        ``-(D log(2 pi) + d^T P d) / 2 + log det(P) / 2`` with ``d = value - loc`` and P the precision matrix.
        """
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = dist = torch.einsum("...i, ...ij, ...j -> ...", diff, self.precision_matrix, diff)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) +0.5 * self.precision_logdet

    def rsample(self, sample_shape=torch.Size()):
        """Draw reparametrised samples of shape ``sample_shape + batch_shape + (D,)``.

        Uses a Cholesky factor of the covariance computed from the precision matrix (not from `precision_logdet`).
        """
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

class CustomTransformedDistribution(TransformedDistribution):
    """TransformedDistribution with an extra `log_prob_onbase` (no Jacobian) and extra arguments for `log_prob`.

    Parameters are those of ``torch.distributions.TransformedDistribution`` (`base_distribution`, `transforms`,
    `validate_args`).
    """

    def log_prob_onbase(self, value, *args, **kwargs):
        """Return the log density of the base distribution at the back-transformed `value`, without Jacobian.

        For `ASinhMultivariateNormal` this is the normal log density of ``arcsinh(value)``, i.e. the likelihood of
        the arcsinh-transformed data rather than of `value` itself.

        Parameters
        ----------
        value : torch.Tensor
            Values on the scale of the transformed distribution.
        *args, **kwargs
            Passed to the base distribution's ``log_prob``.

        Returns
        -------
        torch.Tensor
            Log density, one value per event (the event axes are summed).
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
        """Return the log density at `value`: base log density of the back-transformed value minus log |Jacobian|.

        Same as ``TransformedDistribution.log_prob``, but passes `args` and `kwargs` on to the base distribution's
        ``log_prob``.

        Parameters
        ----------
        value : torch.Tensor
            Values on the scale of the transformed distribution.
        *args, **kwargs
            Passed to the base distribution's ``log_prob``.

        Returns
        -------
        torch.Tensor
            Log density, one value per event.
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
    r"""Elementwise transform :math:`y = \sinh(x)`, inverse :math:`x = \operatorname{arcsinh}(y)`.

    A bijective ``torch.distributions.transforms.Transform``; :math:`dy/dx = \cosh(x) > 0`.
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
        """Return the elementwise ``log |dy/dx| = log cosh(x)`` (`y` is not used)."""
        return x.cosh().log()

class ASinhMultivariateNormal(CustomTransformedDistribution):
    r"""Multivariate normal on the arcsinh scale: :math:`y = \sinh(z)`, :math:`z \sim N(loc, \Sigma)`.

    Modelled after torch's ``LogNormal``. `loc` and the covariance (or precision, scale_tril) describe ``arcsinh(y)``;
    give the location either as `loc` (arcsinh scale) or as `loc_raw` (the scale of `y`, converted with
    ``loc = arcsinh(loc_raw)``, so `loc_raw` is the median of each coordinate). In the rainbow model, `y` is the
    intensity divided by the arcsinh cofactor (plus a shift), so `loc_raw` is a peak position divided by the cofactor.

    With both `precision_matrix` and `precision_logdet` the base distribution is a `CustomMultivariateNormal` that uses
    the given log-determinant; otherwise it is ``torch.distributions.MultivariateNormal`` (then `precision_logdet`, if
    given, is ignored).

    Parameters
    ----------
    loc : torch.Tensor of shape (..., D), optional
        Mean of ``arcsinh(y)``. Exactly one of `loc` and `loc_raw` is required.
    loc_raw : torch.Tensor of shape (..., D), optional
        Location on the scale of `y`.
    covariance_matrix : torch.Tensor of shape (..., D, D), optional
        Covariance of ``arcsinh(y)``. Exactly one of `covariance_matrix`, `precision_matrix` and `scale_tril` is
        required.
    precision_matrix : torch.Tensor of shape (..., D, D), optional
        Precision of ``arcsinh(y)``.
    scale_tril : torch.Tensor of shape (..., D, D), optional
        Lower-triangular Cholesky factor of the covariance of ``arcsinh(y)``.
    precision_logdet : torch.Tensor of shape (...), optional
        ``log det(precision_matrix)``; only used together with `precision_matrix`.
    validate_args : bool, optional
        Passed to the torch distributions.

    Raises
    ------
    ValueError
        If not exactly one of `covariance_matrix`, `precision_matrix`, `scale_tril`, or not exactly one of `loc`,
        `loc_raw` is given.

    Notes
    -----
    `log_prob` includes the Jacobian of the sinh transform, :math:`-\sum_i \tfrac12 \log(1 + y_i^2)`;
    `log_prob_onbase` leaves it out. The mean, mode and variance are not implemented; `median` is ``sinh(loc)``
    (coordinate-wise).
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
        """Return ``"ASinh transformed "`` + the representation of the base distribution."""
        return "ASinh transformed "+self.base_dist.__repr__()

    @property
    def loc(self):
        """Location on the arcsinh scale, shape (..., D)."""
        return self.base_dist.loc

    @property
    def covariance_matrix(self):
        """Covariance on the arcsinh scale, shape (..., D, D); AttributeError with a `CustomMultivariateNormal` base."""
        return self.base_dist.covariance_matrix

    @property
    def mean(self):
        """Not implemented (raises NotImplementedError)."""
        raise NotImplementedError

    @property
    def mode(self):
        """Not implemented (raises NotImplementedError)."""
        raise NotImplementedError

    @property
    def median(self):
        """Coordinate-wise median ``sinh(loc)``, shape (..., D)."""
        return self.base_dist.loc.sinh()

    @property
    def variance(self):
        """Not implemented (raises NotImplementedError)."""
        raise NotImplementedError
