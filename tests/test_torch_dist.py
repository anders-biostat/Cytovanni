"""cytovanni.torch.dist: the arcsinh multivariate normal of the rainbow bead model."""
import math

import pytest
import torch
from torch.distributions import MultivariateNormal

from cytovanni.torch import ASinhMultivariateNormal
from cytovanni.torch.dist import CustomMultivariateNormal, CustomTransformedDistribution, HyperbolicSineTransform

COV = torch.tensor([[1., .3], [.3, .5]])
LOC = torch.tensor([.2, -.4])


def test_custom_mvn_log_prob_matches_torch():
    P = torch.linalg.inv(COV)
    d = CustomMultivariateNormal(LOC, P, torch.logdet(P))
    v = torch.randn(10, 2)
    assert torch.allclose(d.log_prob(v), MultivariateNormal(LOC, COV).log_prob(v), atol=1e-5)


def test_custom_mvn_batch_shape_broadcasts():
    P = torch.eye(2).expand(3, 2, 2)
    d = CustomMultivariateNormal(torch.zeros(2), P, torch.zeros(3))
    assert d.batch_shape == (3,) and d.event_shape == (2,)
    assert d.log_prob(torch.zeros(3, 2)).shape == (3,)


def test_custom_mvn_rsample_shape_and_covariance():
    """R2-D5: rsample honours sample_shape (it returned a single sample)."""
    torch.manual_seed(0)
    P = torch.linalg.inv(COV)
    d = CustomMultivariateNormal(LOC, P, torch.logdet(P))
    s = d.rsample((20000,))
    assert s.shape == (20000, 2)
    assert torch.allclose(s.mean(0), LOC, atol=.03)
    assert torch.allclose(torch.cov(s.T), COV, atol=.03)
    assert d.sample((3, 4)).shape == (3, 4, 2)


def test_sinh_transform():
    t = HyperbolicSineTransform()
    x = torch.tensor([-1., 0., 2.])
    assert torch.allclose(t(x), x.sinh()) and torch.allclose(t.inv(t(x)), x)
    assert torch.allclose(t.log_abs_det_jacobian(x, t(x)), x.cosh().log())
    assert t == HyperbolicSineTransform()


def test_asinh_log_prob_is_normal_of_arcsinh_plus_jacobian():
    d = ASinhMultivariateNormal(loc=LOC, covariance_matrix=COV)
    v = torch.tensor([[.5, -1.], [3., 10.]])
    expected = MultivariateNormal(LOC, COV).log_prob(v.asinh()) - .5 * torch.log1p(v ** 2).sum(-1)
    assert torch.allclose(d.log_prob(v), expected, atol=1e-5)
    assert torch.allclose(d.log_prob_onbase(v), MultivariateNormal(LOC, COV).log_prob(v.asinh()), atol=1e-5)


def test_asinh_loc_raw_is_median():
    d = ASinhMultivariateNormal(loc_raw=torch.tensor([2., 5.]), scale_tril=torch.eye(2))
    assert torch.allclose(d.loc, torch.tensor([2., 5.]).asinh())
    assert torch.allclose(d.median, torch.tensor([2., 5.]))
    torch.manual_seed(0)
    s = d.sample((20000,))
    assert torch.allclose(s.median(0).values, torch.tensor([2., 5.]), rtol=.05)


def test_asinh_precision_parametrisations_agree():
    P = torch.linalg.inv(COV)
    v = torch.randn(5, 2)
    a = ASinhMultivariateNormal(loc=LOC, covariance_matrix=COV).log_prob(v)
    b = ASinhMultivariateNormal(loc=LOC, precision_matrix=P).log_prob(v)
    c = ASinhMultivariateNormal(loc=LOC, precision_matrix=P, precision_logdet=torch.logdet(P))
    assert isinstance(c.base_dist, CustomMultivariateNormal)
    assert isinstance(ASinhMultivariateNormal(loc=LOC, precision_matrix=P).base_dist, MultivariateNormal)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c.log_prob(v), atol=1e-5)


def test_asinh_covariance_matrix_property():
    d = ASinhMultivariateNormal(loc=LOC, covariance_matrix=COV)
    assert torch.allclose(d.covariance_matrix, COV)
    P = torch.linalg.inv(COV)
    custom = ASinhMultivariateNormal(loc=LOC, precision_matrix=P, precision_logdet=torch.logdet(P))
    with pytest.raises(AttributeError):
        custom.covariance_matrix


@pytest.mark.parametrize("kw", [dict(loc=LOC), dict(loc=LOC, covariance_matrix=COV, scale_tril=torch.eye(2)),
                                dict(covariance_matrix=COV), dict(loc=LOC, loc_raw=LOC, covariance_matrix=COV)])
def test_asinh_argument_checks(kw):
    with pytest.raises(ValueError):
        ASinhMultivariateNormal(**kw)


@pytest.mark.parametrize("prop", ["mean", "mode", "variance"])
def test_asinh_not_implemented_moments(prop):
    d = ASinhMultivariateNormal(loc=LOC, covariance_matrix=COV)
    with pytest.raises(NotImplementedError):
        getattr(d, prop)


def test_asinh_repr_and_rsample_shape():
    d = ASinhMultivariateNormal(loc=LOC, covariance_matrix=COV)
    assert repr(d).startswith("ASinh transformed MultivariateNormal")
    assert d.rsample((7,)).shape == (7, 2)
    assert isinstance(d, CustomTransformedDistribution)


def test_asinh_log_prob_gradient_wrt_loc():
    loc = LOC.clone().requires_grad_(True)
    d = ASinhMultivariateNormal(loc=loc, covariance_matrix=COV)
    d.log_prob(torch.tensor([[1., 1.]])).sum().backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()


def test_asinh_density_integrates_to_one_1d():
    d = ASinhMultivariateNormal(loc=torch.tensor([1.]), covariance_matrix=torch.tensor([[.25]]))
    y = torch.linspace(-30, 30, 200001)[:, None]
    p = d.log_prob(y).exp()
    assert math.isclose(torch.trapezoid(p, y[:, 0]).item(), 1., rel_tol=1e-3)
