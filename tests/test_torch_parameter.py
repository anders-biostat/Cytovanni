"""cytovanni.torch.parameter: extendable parameters (per-batch values, frozen on extension)."""
import pytest
import torch
import torch.nn.functional as F

from cytovanni.torch import ExtendableParameter, PrecisionExtendableParameter, CovarianceTrilExtendableParameter


def _step(p, target, lr=.5, n=1):
    """A few SGD steps pulling p() towards `target`."""
    opt = torch.optim.SGD([p.param_trainable], lr=lr)
    for _ in range(n):
        opt.zero_grad()
        ((p() - target) ** 2).sum().backward()
        opt.step()


# ------------------------------------------------------------------ ExtendableParameter
def test_initial_value_broadcast_and_shape():
    p = ExtendableParameter((2, 3), 1, torch.tensor([[1.], [2.]]))
    assert p.shape == (2, 3)
    assert torch.equal(p(), torch.tensor([[1., 1., 1.], [2., 2., 2.]]))
    assert p.param_trainable.requires_grad and not p.param_fixed.requires_grad


def test_initial_value_not_broadcastable_raises():
    with pytest.raises(RuntimeError):
        ExtendableParameter((2, 3), 1, torch.ones(4))


def test_all_parts_in_state_dict():
    p = ExtendableParameter((2,), 0, 0., axis_norm=0)
    assert set(p.state_dict()) == {"param_fixed", "param_mask_trainable", "param_trainable", "param_mask_meannorm"}


def test_docstring_example_extend_shape():
    p = ExtendableParameter((3,), 0, 0., axis_norm=0)
    p.extend([3, 4])
    assert p().shape == torch.Size([5])


def test_axis_norm_subtracts_mean():
    p = ExtendableParameter((2, 3), 1, 0., axis_norm=1)
    p.param_trainable.data = torch.tensor([[1., 2., 3.], [0., 0., 6.]])
    assert torch.allclose(p(), torch.tensor([[-1., 0., 1.], [-2., -2., 4.]]))
    assert torch.allclose(p().mean(1), torch.zeros(2))


def test_extend_freezes_old_entries_and_trains_new_ones():
    p = ExtendableParameter((2,), 0, 1.)
    _step(p, torch.tensor([3., 3.]), n=20)
    before = p().detach().clone()
    p.extend([2, 3])
    assert p.shape == (4,)
    assert torch.allclose(p()[:2], before)
    assert torch.equal(p()[2:].detach(), torch.tensor([1., 1.]))  # new entries start at init_value
    assert p.param_mask_trainable.tolist() == [0., 0., 1., 1.]
    _step(p, torch.tensor([-5., -5., -5., -5.]), n=5)
    assert torch.allclose(p()[:2], before)  # frozen
    assert (p()[2:] < 0).all()


def test_extend_with_existing_indices_only_freezes():
    p = ExtendableParameter((3,), 0, 2.)
    p.extend([1])
    assert p.shape == (3,) and p.param_mask_trainable.sum() == 0
    assert torch.equal(p(), torch.full((3,), 2.))


def test_freeze_moves_everything_to_fixed():
    p = ExtendableParameter((2, 2), 0, .5)
    p.freeze()
    assert p.param_mask_trainable.sum() == 0
    assert torch.equal(p.param_fixed, torch.full((2, 2), .5))
    assert torch.equal(p(), torch.full((2, 2), .5))


def test_extend_meannorm_excludes_new_entries_by_default():
    """The zero point stays as fitted: new entries do not enter the mean."""
    p = ExtendableParameter((3,), 0, 0., axis_norm=0)
    p.param_trainable.data = torch.tensor([1., 2., 3.])
    old = p().detach().clone()
    p.extend([3])
    p.param_trainable.data[3] = 10.
    assert torch.allclose(p()[:3], old)
    assert p.param_mask_meannorm.tolist() == [1., 1., 1., 0.]


def test_extend_keep_prior_structure():
    p = ExtendableParameter((2,), 0, 0., axis_norm=0)
    p.freeze_selective([0])
    p.extend([2], keep_prior_trainable_structure=True)
    assert p.param_mask_trainable.tolist() == [0., 1., 1.]
    assert p.param_mask_meannorm.tolist() == [1., 1., 1.]


def test_freeze_selective_keeps_value_and_stops_training():
    p = ExtendableParameter((2, 3), 1, 0.)
    p.param_trainable.data = torch.arange(6.).view(2, 3)
    p.freeze_selective((slice(None), [0, 1]))
    assert torch.equal(p(), torch.arange(6.).view(2, 3))
    assert p.param_mask_trainable.tolist() == [[0., 0., 1.], [0., 0., 1.]]
    _step(p, torch.zeros(2, 3), n=3)
    assert torch.equal(p()[:, :2].detach(), torch.tensor([[0., 1.], [3., 4.]]))


def test_set_trainable_part_only_overwrites_trainable_entries():
    p = ExtendableParameter((4,), 0, 0.)
    p.freeze_selective([0])
    p.param_fixed.data[0] = 7.
    p.set_trainable_part(torch.tensor([9., 9., 9., 9.]))
    assert p().tolist() == [7., 9., 9., 9.]
    p.set_trainable_part(torch.tensor([1., 2.]), sl=[2, 3])
    assert p().tolist() == [7., 9., 1., 2.]


def test_add_to_fixed():
    p = ExtendableParameter((2, 2), 0, 0.)
    p.add_to_fixed(torch.tensor([1., 2.]))
    assert torch.equal(p(), torch.tensor([[1., 2.], [1., 2.]]))
    p.add_to_fixed(torch.tensor(5.), sl=(0, 0))
    assert p()[0, 0] == 6.


def test_requires_grad_returns_self_and_respects_allow_grad():
    """R2-E1: requires_grad_ returns self (torch convention)."""
    p = ExtendableParameter((2,), 0, 0.)
    assert p.requires_grad_(False) is p and not p.param_trainable.requires_grad
    p.requires_grad_(True)
    assert p.param_trainable.requires_grad and not p.param_fixed.requires_grad
    q = ExtendableParameter((2,), 0, 0., allow_grad=False, axis_norm=0)
    assert q.requires_grad_(True) is q and not q.param_trainable.requires_grad


def test_extend_keeps_requires_grad():
    p = ExtendableParameter((2,), 0, 0.)
    p.requires_grad_(False)
    p.extend([3])
    assert not p.param_trainable.requires_grad


def test_state_dict_roundtrip_after_extend():
    p = ExtendableParameter((2,), 0, 1., axis_norm=0)
    p.extend([2])
    p.param_trainable.data[2] = 4.
    q = ExtendableParameter((2,), 0, 1., axis_norm=0)
    q.extend([2])
    q.load_state_dict(p.state_dict())
    assert torch.equal(q(), p())


def test_device_and_repr(device):
    p = ExtendableParameter((2,), 0, 0., axis_norm=0).to(device)
    assert p.device.type == device
    p.extend([3])
    assert p.device.type == device and p().device.type == device
    assert "Extendable Parameter of shape (4,)" in repr(p) and "Mean Norm Mask" in repr(p)


# ------------------------------------------------------------------ PrecisionExtendableParameter
def test_precision_initial_value_and_logdet():
    p = PrecisionExtendableParameter((3, 2, 2), 0, 2.)
    prec, logdet = p()
    assert torch.allclose(prec, 4 * torch.eye(2).expand(3, 2, 2))
    assert torch.allclose(logdet, torch.full((3,), 2 * torch.log(torch.tensor(4.)).item()))


def test_precision_logdet_matches_direct_for_random_factor():
    p = PrecisionExtendableParameter((2, 3, 3), 0, 1.)
    p.param_trainable.data = torch.randn(2, 3, 3) + 2 * torch.eye(3)
    prec, logdet = p()
    assert torch.allclose(prec, p.param_trainable @ p.param_trainable.transpose(-1, -2))
    assert torch.allclose(logdet, torch.logdet(prec), atol=1e-4)


def test_precision_negative_determinant_falls_back():
    p = PrecisionExtendableParameter((2, 2), 0, 1.)
    p.param_trainable.data = torch.tensor([[-1., 0.], [0., 2.]])
    prec, logdet = p()
    assert torch.isclose(logdet, torch.log(torch.tensor(4.)))


def test_precision_singular_factor_gives_minus_inf_logdet():
    """A singular A gives logdet -inf (not NaN), so no ValueError is raised; see findings (Questions / R5)."""
    p = PrecisionExtendableParameter((2, 2), 0, 1.)
    p.param_trainable.data = torch.zeros(2, 2)
    logdet = p()[1]
    assert torch.isinf(logdet) and logdet < 0


def test_precision_one_axis_raises():
    with pytest.raises(ValueError):
        PrecisionExtendableParameter((3,), 0, 1.)


def test_precision_extend_adds_identity_blocks(device):
    p = PrecisionExtendableParameter((1, 2, 2), 0, 3.).to(device)
    p.extend([2])
    prec, _ = p()
    assert prec.shape == (3, 2, 2) and prec.device.type == device
    assert torch.allclose(prec.cpu(), 9 * torch.eye(2).expand(3, 2, 2))


# ------------------------------------------------------------------ CovarianceTrilExtendableParameter
def test_covtril_initial_diagonal_is_softplus_of_init():
    c = CovarianceTrilExtendableParameter((2, 2), 0, .2)
    L = c()
    assert torch.allclose(torch.diag(L), F.softplus(torch.tensor(.2)).expand(2))
    assert L[0, 1] == 0 and L[1, 0] == 0


def test_covtril_lower_triangular_and_positive_diagonal():
    c = CovarianceTrilExtendableParameter((4, 3, 3), 0, .2)
    c.param_trainable.data = torch.randn(4, 3, 3) - 3
    L = c()
    assert torch.equal(L, torch.tril(L))
    assert (torch.diagonal(L, dim1=-2, dim2=-1) > 0).all()
    assert c.param_mask_trainable[0].tolist() == [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]


def test_covtril_gradient_flows():
    c = CovarianceTrilExtendableParameter((2, 2), 0, .2)
    c().sum().backward()
    g = c.param_trainable.grad
    assert g[0, 1] == 0 and g[1, 0] == 1 and 0 < g[0, 0] < 1


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason="documented ValueError, but the trainable mask (torch.tril) raises a RuntimeError first")
def test_covtril_one_axis_raises():
    with pytest.raises(ValueError):
        CovarianceTrilExtendableParameter((3,), 0, .2)


def test_covtril_extend_on_device(device):
    """R3b-B3: the init helpers create tensors on the parameter's device (extend on the GPU crashed)."""
    c = CovarianceTrilExtendableParameter((2, 2, 2), 0, .2).to(device)
    c.extend([3])
    assert c.shape == (4, 2, 2) and c().device.type == device
