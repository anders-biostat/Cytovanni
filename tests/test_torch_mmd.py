"""cytovanni.torch.mmd: MMD loss with a multi-scale RBF kernel and event weights."""
import numpy as np
import pytest
import torch

from cytovanni.torch import MMDLoss, subsamplemean_MMD
from cytovanni.torch.mmd import RBF, RBF_batched, MMDLoss_batched, batched_cdist, _bandwidth_floor


def _samples(seed=0, shift=1.):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(60, 2, generator=g), torch.randn(80, 2, generator=g) + shift


def test_rbf_multipliers_and_kernel_formula():
    k = RBF(bandwidth=2.)
    assert torch.allclose(k.bandwidth_multipliers, torch.tensor([.25, .5, 1., 2., 4.]))
    X = torch.tensor([[0., 0.], [1., 1.]])
    K = k(X)
    d2 = 2.
    expected = sum(np.exp(-d2 / (2. * m)) for m in [.25, .5, 1., 2., 4.])
    assert K.shape == (2, 2)
    assert np.isclose(K[0, 1].item(), expected, rtol=1e-5) and np.isclose(K[0, 0].item(), 5.)


def test_estimate_bandwidth_mean_offdiagonal_squared_distance():
    X = torch.tensor([[0., 0.], [3., 4.]])
    assert np.isclose(RBF.estimate_bandwidth(X).item(), 25., rtol=1e-5)
    assert np.isclose(MMDLoss.estimate_bandwidth(X).item(), 25., rtol=1e-5)


def test_mmd_zero_for_identical_samples_positive_otherwise():
    X, Y = _samples()
    L = MMDLoss()
    assert abs(L(X, X).item()) < 1e-5
    assert L(X, Y).item() > L(X, _samples(seed=1, shift=0.)[0]).item() > -1e-6


def test_mmd_increases_with_shift():
    X, _ = _samples()
    L = MMDLoss(bandwidth=2.)
    vals = [L(X, X + s).item() for s in [0., .5, 1., 2.]]
    assert vals == sorted(vals) and vals[0] < 1e-6


def test_mmd_biased_vstatistic_formula():
    X, Y = _samples()
    L = MMDLoss(bandwidth=3.)
    k = RBF(bandwidth=3.)
    expected = k(X).mean() - 2 * torch.cat([k(torch.vstack([X, Y]))[:60, 60:]]).mean() + k(Y).mean()
    assert torch.isclose(L(X, Y), expected, atol=1e-5)


def test_mmd_uniform_weights_equal_unweighted():
    X, Y = _samples()
    L = MMDLoss(bandwidth=2.)
    ref = L(X, Y)
    assert torch.isclose(L(X, Y, torch.full((60,), 3.), None), ref, atol=1e-5)
    assert torch.isclose(L(X, Y, torch.ones(60), torch.full((80,), .1)), ref, atol=1e-5)


def test_mmd_weights_change_loss_like_resampling():
    """Weight 2 on an event acts like duplicating it."""
    X, Y = _samples()
    L = MMDLoss(bandwidth=2.)
    w = torch.ones(60); w[:10] = 2.
    dup = torch.vstack([X, X[:10]])
    assert torch.isclose(L(X, Y, w, None), L(dup, Y), atol=1e-5)


@pytest.mark.parametrize("w", [torch.zeros(60), torch.full((60,), float("nan"))])
def test_mmd_bad_weight_sums_raise(w):
    """R2b-G2-14: weights with a zero or non-finite sum raise instead of giving a silent NaN loss."""
    X, Y = _samples()
    with pytest.raises(ValueError):
        MMDLoss()(X, Y, w, None)
    with pytest.raises(ValueError):
        MMDLoss()(X, Y, None, w[:1].expand(80).clone())


def test_mmd_identical_points_finite():
    """R2b-G2-21: automatic bandwidth on identical points is floored, the loss is 0 instead of NaN."""
    X = torch.ones(10, 3)
    assert RBF.estimate_bandwidth(X).item() > 0
    v = MMDLoss()(X, X.clone())
    assert torch.isfinite(v) and abs(v.item()) < 1e-6


def test_bandwidth_floor_only_affects_tiny_values():
    assert _bandwidth_floor(torch.tensor(0.)).item() > 0
    assert _bandwidth_floor(torch.tensor(2.5)).item() == 2.5
    assert _bandwidth_floor(torch.tensor(0., dtype=torch.float64)).item() >= torch.finfo(torch.float32).tiny ** .5


def test_mmd_gradient_wrt_sample():
    X, Y = _samples()
    X = X.clone().requires_grad_(True)
    MMDLoss(bandwidth=2.)(X, Y).backward()
    assert torch.isfinite(X.grad).all() and X.grad.abs().sum() > 0


def test_mmd_on_device(device):
    X, Y = _samples()
    L = MMDLoss(bandwidth=2.).to(device)
    v = L(X.to(device), Y.to(device))
    assert v.device.type == device
    assert np.isclose(v.item(), MMDLoss(bandwidth=2.)(X, Y).item(), rtol=1e-4)


def test_subsamplemean_mmd_mean_and_sem():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(500, 2)), rng.normal(size=(500, 2))
    m, sem = subsamplemean_MMD(x, y, M=5, N=100)
    m_shift, _ = subsamplemean_MMD(x, y + 2, M=5, N=100)
    assert sem >= 0 and m_shift > m + 5 * sem
    # N larger than the samples: all events used, every repetition identical
    m_all, sem_all = subsamplemean_MMD(x[:50], y[:50], mmd=MMDLoss(bandwidth=1.), M=3, N=1000)
    assert sem_all < 1e-6
    assert np.isclose(m_all, MMDLoss(bandwidth=1.)(torch.tensor(x[:50], dtype=torch.float32),
                                                   torch.tensor(y[:50], dtype=torch.float32)).item(), rtol=1e-5)


def test_batched_cdist_matches_torch_cdist():
    X, Y = torch.randn(3, 5, 2), torch.randn(3, 4, 2)
    assert torch.allclose(batched_cdist(X, Y), torch.cdist(X, Y), atol=1e-5)


def test_rbf_batched_matches_rbf_per_batch():
    X = torch.randn(3, 6, 2)
    Kb = RBF_batched()(X)
    assert Kb.shape == (3, 6, 6)
    for i in range(3):
        assert torch.allclose(Kb[i], RBF()(X[i]), atol=1e-5)
    bw = RBF_batched.estimate_bandwidth(X)
    assert bw.shape == (3, 1, 1)
    assert torch.allclose(bw[1, 0, 0], RBF.estimate_bandwidth(X[1]), rtol=1e-5)
    assert torch.allclose(RBF_batched(bandwidth=torch.full((3, 1, 1), 2.))(X)[0], RBF(bandwidth=2.)(X[0]), atol=1e-5)


def test_rbf_batched_float_bandwidth():
    X = torch.randn(3, 6, 2)
    assert torch.allclose(RBF_batched(bandwidth=2.)(X)[0], RBF(bandwidth=2.)(X[0]), atol=1e-5)


def test_mmdloss_batched_not_implemented():
    with pytest.raises(NotImplementedError):
        MMDLoss_batched()
