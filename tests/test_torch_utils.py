"""cytovanni.torch.utils: straight-through clamp, NaN-safe NLL, weight checks, memory diagnostics."""
import pytest
import torch

from cytovanni.torch import diffclamp, _ensure_real_nll, torch_print_memory, torch_show_tensors_inmemory
from cytovanni.torch.utils import _check_weight_sums, DifferentiableClamp


def test_diffclamp_clamps_values():
    z = torch.tensor([-2., .5, 9.])
    assert torch.equal(diffclamp(z, 0, 1), torch.tensor([0., .5, 1.]))
    assert torch.equal(diffclamp(z, None, 1), torch.tensor([-2., .5, 1.]))
    assert torch.equal(diffclamp(z, 0, None), torch.tensor([0., .5, 9.]))


def test_diffclamp_gradient_is_one_everywhere():
    z = torch.tensor([-2., .5, 9.], requires_grad=True)
    (diffclamp(z, 0, 1) * torch.tensor([1., 2., 3.])).sum().backward()
    assert torch.equal(z.grad, torch.tensor([1., 2., 3.]))
    # plain clamp for comparison: no gradient where clipped
    z2 = z.detach().clone().requires_grad_(True)
    z2.clamp(0, 1).sum().backward()
    assert torch.equal(z2.grad, torch.tensor([0., 1., 0.]))


def test_differentiable_clamp_function_is_what_diffclamp_uses():
    z = torch.tensor([-1., 2.])
    assert torch.equal(DifferentiableClamp.apply(z, 0., 1.), diffclamp(z, 0., 1.))


def test_ensure_real_nll_clips_and_removes_nan():
    nll = torch.tensor([-1., 5., 2e4, float("nan"), float("inf"), float("-inf")])
    out = _ensure_real_nll(nll)
    # inf is clamped to 1e4 before nan_to_num; -inf to 0
    assert out.tolist() == [0., 5., 1e4, 0., 1e4, 0.]


@pytest.mark.parametrize("w", [torch.zeros(3), torch.tensor([1., float("nan"), 1.]), torch.tensor([1., float("inf")]),
                               torch.tensor([-1., -2.])])
def test_check_weight_sums_rejects_bad_sums(w):
    with pytest.raises(ValueError):
        _check_weight_sums(w)


def test_check_weight_sums_accepts_positive_rows():
    _check_weight_sums(torch.tensor([[1., 0.], [0., 2.]]))
    with pytest.raises(ValueError):
        _check_weight_sums(torch.tensor([[1., 0.], [0., 0.]]))  # one bad row of a batch


@pytest.mark.filterwarnings("ignore::FutureWarning")  # hasattr(obj, "data") touches deprecated torch attributes
def test_show_tensors_inmemory_prints_sizes(capsys):
    keep = torch.zeros(17, 3)  # noqa: F841
    torch_show_tensors_inmemory()
    assert "torch.Size([17, 3])" in capsys.readouterr().out


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_print_memory_reports_three_lines(capsys):
    torch_print_memory()
    out = capsys.readouterr().out
    assert out.count("GB") == 3 and "memory_allocated" in out
