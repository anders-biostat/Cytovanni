"""cytovanni.torch.unmx: inversion of (stain, channel) spectra and differentiable unmixing."""
import numpy as np
import pytest
import torch

import synthetic
from cytovanni.exceptions import SpilloverInversionException
from cytovanni.torch import torch_invert_spectra, torch_apply_unmixing_inv, torch_apply_unmixing_lstsq


def _spectra(n_stain, n_channel, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_stain, n_channel, generator=g) + torch.eye(n_stain, n_channel)


def test_invert_square_is_matrix_inverse():
    S = torch.tensor(synthetic.SPECTRA.to_numpy(), dtype=torch.float32)
    inv = torch_invert_spectra(S)
    assert inv.shape == (3, 3)
    assert torch.allclose(S @ inv, torch.eye(3), atol=1e-5)


def test_invert_spectral_is_right_inverse_and_shape_channel_stain():
    S = _spectra(3, 6)
    inv = torch_invert_spectra(S)
    assert inv.shape == (6, 3)  # (channel, stain)
    assert torch.allclose(S @ inv, torch.eye(3), atol=1e-5)
    assert torch.allclose(inv, S.T @ torch.linalg.inv(S @ S.T), atol=1e-6)


def test_invert_pinv_matches_explicit():
    S = _spectra(3, 6)
    assert torch.allclose(torch_invert_spectra(S, use_pinv=True), torch_invert_spectra(S), atol=1e-4)


def test_invert_fewer_channels_than_stains_raises():
    with pytest.raises(SpilloverInversionException):
        torch_invert_spectra(_spectra(4, 3))


def test_invert_batched_leading_axes():
    S = torch.stack([_spectra(2, 4, seed=i) for i in range(5)])  # (5, stain, channel)
    inv = torch_invert_spectra(S)
    assert inv.shape == (5, 4, 2)
    for i in range(5):
        assert torch.allclose(inv[i], torch_invert_spectra(S[i]), atol=1e-6)


def test_unmixing_inv_recovers_abundances(device):
    S = _spectra(3, 5).to(device)
    a = torch.randn(20, 3, device=device) * 100
    x = a @ S
    out = torch_apply_unmixing_inv(torch_invert_spectra(S), x)
    assert out.shape == (20, 3)
    assert torch.allclose(out, a, atol=1e-2)


def test_unmixing_inv_one_matrix_per_event():
    S = torch.stack([_spectra(2, 3, seed=i) for i in range(4)])  # (events, stain, channel)
    a = torch.randn(4, 2)
    x = torch.einsum("es, esc -> ec", a, S)
    out = torch_apply_unmixing_inv(torch_invert_spectra(S), x)
    assert torch.allclose(out, a, atol=1e-4)


def test_unmixing_inv_is_differentiable_wrt_spectra():
    S = _spectra(2, 3).requires_grad_(True)
    x = torch.rand(5, 3)
    torch_apply_unmixing_inv(torch_invert_spectra(S), x).sum().backward()
    assert S.grad is not None and torch.isfinite(S.grad).all()


def test_lstsq_matches_inverse_with_broadcast_leading_axis():
    S = _spectra(3, 5)
    a = torch.randn(7, 3)
    x = a @ S
    assert torch.allclose(torch_apply_unmixing_lstsq(S[None], x), a, atol=1e-4)
    assert torch.allclose(torch_apply_unmixing_lstsq(S.expand(7, 3, 5), x), a, atol=1e-4)


def test_lstsq_single_matrix_without_event_axis_raises():
    """Documented: a single (stain, channel) matrix for x of shape (events, channel) raises a RuntimeError."""
    S = _spectra(3, 5)
    with pytest.raises(RuntimeError):
        torch_apply_unmixing_lstsq(S, torch.randn(7, 5))
