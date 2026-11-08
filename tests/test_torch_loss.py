"""cytovanni.torch.loss: LossContainer (named scalar losses, N-weighted averaging)."""
import pytest
import torch

from cytovanni.torch import LossContainer
from cytovanni.torch.loss import is_LossContainer


def test_values_become_0dim_tensors_mean_taken():
    d = {"a": torch.tensor([1., 3.]), "b": 2.}
    lc = LossContainer(d, N=2)
    assert lc["a"].dim() == 0 and lc["a"].item() == 2.
    assert isinstance(lc["b"], torch.Tensor) and lc["b"].item() == 2.
    assert lc.losses is d  # used directly, converted in place
    assert lc.N == 2


def test_docstring_example_weighted_add():
    lc = LossContainer({"ddist": torch.tensor([1., 3.])}, N=2)
    tot = lc + LossContainer({"ddist": 4.}, N=6)
    assert torch.isclose(tot["ddist"], torch.tensor(3.5))
    assert tot.N == 8


def test_add_detaches():
    x = torch.tensor(1., requires_grad=True)
    tot = LossContainer({"a": x * 2}) + LossContainer({"a": x * 3})
    assert not tot["a"].requires_grad


def test_sum_with_none_start_works_builtin_sum_raises():
    cs = [LossContainer({"a": float(i)}, N=1) for i in range(4)]
    assert sum(cs, None)["a"].item() == pytest.approx(1.5)
    with pytest.raises(NotImplementedError):
        sum(cs)
    assert (cs[0] + None) is cs[0]


def test_add_with_different_keys_raises():
    with pytest.raises(ValueError):
        LossContainer({"a": 1.}) + LossContainer({"b": 1.})


def test_mul_by_number_returns_scaled_copy():
    lc = LossContainer({"a": 2., "b": 3.})
    res = 2 * lc
    assert res["a"].item() == 4. and res["b"].item() == 6.
    assert lc["a"].item() == 2.  # unchanged
    assert (lc * 0.5)["b"].item() == 1.5


def test_mul_by_dict():
    lc = LossContainer({"a": 2., "b": 3.})
    res = lc * {"a": 10., "b": 0.}
    assert res["a"].item() == 20. and res["b"].item() == 0.
    with pytest.raises(ValueError):
        lc * {"a": 1.}
    part = LossContainer({"a": 2., "b": 3.}, allow_partialdictmult=True) * {"a": 10.}
    assert part["a"].item() == 20. and part["b"].item() == 3.


@pytest.mark.parametrize("bad", [None, [1., 2.]])
def test_mul_unsupported_raises_notimplemented(bad):
    with pytest.raises(NotImplementedError):
        LossContainer({"a": 1.}) * bad


def test_getitem_list_sums():
    lc = LossContainer({"a": 1., "b": 2., "c": 4.})
    assert lc[["a", "c"]].item() == 5.


def test_setitem_converts():
    lc = LossContainer({"a": 1.})
    lc["b"] = torch.tensor([2., 4.])
    assert lc["b"].dim() == 0 and lc["b"].item() == 3.
    lc["c"] = 7
    assert isinstance(lc["c"], torch.Tensor)


def test_update():
    lc = LossContainer({"a": 1.}, N=3)
    lc.update({"b": [1., 3.]})
    assert lc["b"].item() == 2.
    lc.update(None)
    with pytest.raises(ValueError):
        lc.update({"a": 2.})
    with pytest.raises(ValueError):
        lc.update(LossContainer({"c": 1.}, N=4))
    lc.update(LossContainer({"c": 1.}, N=3))
    assert set(lc.keys()) == {"a", "b", "c"}


def test_copy_detach_numberdict():
    x = torch.tensor(2., requires_grad=True)
    lc = LossContainer({"a": x * 3}, N=5)
    cp = lc.copy()
    assert cp.N == 5 and cp.losses is not lc.losses
    lc.detach()
    assert not lc["a"].requires_grad and cp["a"].requires_grad
    assert lc.to_numberdict() == {"a": 6.}


def test_to_module_log():
    calls = []

    class M:
        def log(self, name, value, batch_size):
            calls.append((name, float(value), batch_size))

    LossContainer({"a": 1., "b": 2.}, N=7).to_module_log(M(), prefix="train_")
    assert calls == [("train_a", 1., 7), ("train_b", 2., 7)]


def test_is_losscontainer_and_repr():
    lc = LossContainer({"a": 1.})
    assert is_LossContainer(lc) and not is_LossContainer({"a": 1.})
    assert repr(lc) == "LossContainer with 1 losses."
