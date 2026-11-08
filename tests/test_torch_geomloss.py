"""cytovanni.torch.geomloss: Sinkhorn / Hausdorff divergence between point clouds (tensorized geomloss)."""
import numpy as np
import pytest
import torch

pytest.importorskip("geomloss")
from cytovanni.torch import GeomSampleLoss  # noqa: E402


def _clouds(seed=0, n=200, m=300, shift=.5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=g), torch.randn(m, 3, generator=g) + shift


def test_sinkhorn_zero_for_identical_and_grows_with_shift():
    X, _ = _clouds()
    L = GeomSampleLoss("sinkhorn")
    assert abs(L(X, X).item()) < 1e-4
    vals = [L(X, X + s).item() for s in [.25, .5, 1.]]
    assert vals == sorted(vals) and vals[0] > 0


def test_unknown_mode_raises_notimplemented():
    """R2-K1 (geomloss): unknown modes raise NotImplementedError (was `raise NotImplemented(...)`, a TypeError)."""
    with pytest.raises(NotImplementedError):
        GeomSampleLoss("energy")


def test_hausdorff_constructs_and_runs():
    """R2-G1: the hausdorff mode could not be constructed (AttributeError)."""
    X, Y = _clouds()
    L = GeomSampleLoss("hausdorff", reach=5.)
    v = L(X, Y)
    assert v.dim() == 0 and torch.isfinite(v) and v.item() > abs(L(X, X).item())


def test_weights_are_normalised_per_side():
    """R2-A5: weights summing to N (as from RareMarkerWeights) give the unweighted loss (they gave ~70 instead of 0.13)."""
    X, Y = _clouds()
    L = GeomSampleLoss("sinkhorn")
    ref = L(X, Y)
    assert torch.isclose(L(X, Y, torch.ones(200), torch.ones(300)), ref, rtol=1e-4)
    assert torch.isclose(L(X, Y, torch.full((200,), 1 / 200), torch.full((300,), 1 / 300)), ref, rtol=1e-4)


def test_one_sided_weights_are_used():
    """R2-A5: a single weighted side is no longer dropped; the other side gets uniform weights."""
    X, Y = _clouds()
    L = GeomSampleLoss("sinkhorn")
    w = torch.ones(200); w[X[:, 0] > 0] = 5.
    assert torch.isclose(L(X, Y, torch.ones(200) * 3, None), L(X, Y), rtol=1e-4)
    assert not torch.isclose(L(X, Y, w, None), L(X, Y), rtol=1e-2)
    assert torch.isclose(L(X, Y, w, None), L(X, Y, w, torch.ones(300)), rtol=1e-4)


def test_weights_like_resampling():
    """Weight 2 on an event acts like duplicating it (after normalisation)."""
    X, Y = _clouds(n=50, m=60)
    L = GeomSampleLoss("sinkhorn")
    w = torch.ones(50); w[:10] = 2.
    assert torch.isclose(L(X, Y, w, None), L(torch.vstack([X, X[:10]]), Y), rtol=1e-3)


@pytest.mark.parametrize("w", [torch.zeros(200), torch.full((200,), float("inf"))])
def test_bad_weight_sums_raise(w):
    """R2b-G2-14: zero / non-finite weight sums raise instead of a silent NaN."""
    X, Y = _clouds()
    with pytest.raises(ValueError):
        GeomSampleLoss("sinkhorn")(X, Y, w, None)


def test_batched_input_and_weights():
    X, Y = _clouds(n=40, m=50)
    L = GeomSampleLoss("sinkhorn")
    same = L(torch.stack([X, X]), torch.stack([Y, Y]))
    assert same.shape == (2,) and torch.allclose(same, L(X, Y).expand(2), rtol=1e-4)
    Xb, Yb = torch.stack([X, X + 1]), torch.stack([Y, Y])
    out = L(Xb, Yb)
    # geomloss's eps-scaling schedule uses the diameter of the whole batch, so a batch row is only approximately the
    # unbatched loss (0.3288 vs 0.3270 here); see findings, Questions / R5
    assert torch.isclose(out[0], L(X, Y), rtol=2e-2)
    wb = torch.ones(2, 40) * torch.tensor([[1.], [7.]])
    assert torch.allclose(L(Xb, Yb, wb, None), out, rtol=1e-4)
    bad = wb.clone(); bad[1] = 0.
    with pytest.raises(ValueError):
        L(Xb, Yb, bad, None)


def test_gradient_wrt_points(device):
    X, Y = _clouds()
    X = X.to(device).requires_grad_(True)
    L = GeomSampleLoss("sinkhorn")
    v = L(X, Y.to(device))
    v.backward()
    assert v.device.type == device and torch.isfinite(X.grad).all()


def test_recovers_shift_by_gradient_descent():
    """C: minimising the Sinkhorn loss over a translation recovers the true shift."""
    torch.manual_seed(0)
    Y = torch.randn(300, 2) * .5 + torch.tensor([1.5, -1.])
    X = torch.randn(300, 2) * .5
    shift = torch.zeros(2, requires_grad=True)
    L = GeomSampleLoss("sinkhorn", blur=.1, reach=None)
    opt = torch.optim.Adam([shift], lr=.1)
    for _ in range(80):
        opt.zero_grad(); L(X + shift, Y).backward(); opt.step()
    assert np.allclose(shift.detach().numpy(), [1.5, -1.], atol=.1)
