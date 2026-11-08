"""Tests of cytovanni.utils.oe (CustomOrdinalEncoder) and cytovanni.utils.rate (event rate)."""
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytovanni.utils import CustomOrdinalEncoder, EventRateEstimator, estimate_event_rate


# ------------------------------------------------------------------ CustomOrdinalEncoder
def test_from_labels_sorted_string_codes():
    oe = CustomOrdinalEncoder.from_labels(["b", "a", "b"])  # docstring example
    assert list(oe.transform(["a", "b"])) == [0, 1]
    assert oe.num_labels == 2 == len(oe) and list(oe.labels) == ["a", "b"] and not oe.allow_unknown
    num = CustomOrdinalEncoder.from_labels([3, 10, 1, 3])
    assert list(num.labels) == ["1", "10", "3"]  # str order: 10 before 3
    assert num.transform(3) == num.transform("3") == 2  # 3 and "3" are the same label


def test_transform_scalar_and_array():
    oe = CustomOrdinalEncoder.from_labels(["x", "y", "z"])
    t = oe.transform("y")
    assert np.ndim(t) == 0 and int(t) == 1
    arr = oe.transform(np.array(["z", "x"]))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,) and arr.dtype.kind == "i"
    assert list(oe.transform(pd.Series(["z", "z"]))) == [2, 2]


def test_inverse_transform_returns_strings():
    oe = CustomOrdinalEncoder.from_labels([1, 2, 3])
    inv = oe.inverse_transform([2, 0])
    assert list(inv) == ["3", "1"] and isinstance(inv, np.ndarray)
    assert oe.inverse_transform(1) == "2" and isinstance(oe.inverse_transform(1), str)


def test_unknown_labels():
    oe = CustomOrdinalEncoder.from_labels(["a", "b"])
    with pytest.raises(ValueError):
        oe.transform(["c"])
    oe2 = CustomOrdinalEncoder.from_labels(["a", "b"], allow_unknown=True)
    assert list(oe2.transform(["b", "c"])) == [1, -1]
    assert oe2.inverse_transform([-1])[0] is None


def test_from_old_keeps_order():
    oe = CustomOrdinalEncoder.from_old(["z", "a", 5])
    assert list(oe.labels) == ["z", "a", "5"]
    assert list(oe.transform(["a", "z", 5])) == [1, 0, 2]
    copy = CustomOrdinalEncoder.from_old(oe.labels)
    assert list(copy.transform(["5", "z"])) == [2, 0]


def test_fit_extend_keeps_codes_and_appends_sorted():
    oe = CustomOrdinalEncoder.from_labels(["b", "a"])
    assert oe.fit_extend(["c"]) == [2]  # docstring example
    assert oe.fit_extend(["zz", "b", "d", "zz"]) == [3, 4]
    assert list(oe.labels) == ["a", "b", "c", "d", "zz"]
    assert list(oe.transform(["a", "b", "c", "d", "zz"])) == [0, 1, 2, 3, 4]
    assert oe.fit_extend(["a"]) == [] and len(oe) == 5


def test_fit_extend_keeps_allow_unknown():
    oe = CustomOrdinalEncoder.from_old(["q"], allow_unknown=True)
    oe.fit_extend(["r"])
    assert oe.allow_unknown and list(oe.transform(["r", "s"])) == [1, -1]


# ------------------------------------------------------------------ EventRateEstimator
def test_event_rate_recovers_poisson_rate(rng):
    t = np.cumsum(rng.exponential(1 / 50., size=20000))
    est = EventRateEstimator(rng.permutation(t))  # any order
    assert est.rate == est.kappa == pytest.approx(50, rel=.03)
    assert est.N == 20000 and est.timediff.shape == (19999,)
    assert np.all(np.diff(est.time) >= 0) and np.all(est.timediff >= 0)
    assert est.kappa == pytest.approx(np.log(2) / np.median(np.diff(np.sort(t))))


def test_event_rate_robust_to_pause(rng):
    t = np.cumsum(rng.exponential(1 / 50., size=20000))
    t[10000:] += 400.  # a 400 s pause: the mean-based estimate halves, the median-based one does not
    assert EventRateEstimator(t).rate == pytest.approx(50, rel=.03)
    assert (len(t) - 1) / (t[-1] - t[0]) < 30


def test_estimate_event_rate_from_obs(rng):
    t = np.cumsum(rng.exponential(1 / 20., size=5000))
    a = anndata.AnnData(np.zeros((5000, 1), np.float32), obs=pd.DataFrame({"Time": t, "T2": 2 * t}, index=[str(i) for i in range(5000)]))
    assert estimate_event_rate(a) == pytest.approx(EventRateEstimator(t).rate)
    assert estimate_event_rate(a, key_time="T2") == pytest.approx(EventRateEstimator(t).rate / 2)


def test_event_rate_plot(rng):
    est = EventRateEstimator(np.cumsum(rng.exponential(1 / 50., size=2000)))
    n_before = len(plt.get_fignums())
    assert est.plot() is None
    assert len(plt.get_fignums()) == n_before + 1
    ax = plt.gcf().axes
    assert len(ax) == 2 and ax[1].get_yscale() == "log"


def test_event_rate_with_quantised_times(rng):
    """Times stored with a finite resolution ($TIMESTEP, here 0.01 s) at 200 events/s: most differences are 0."""
    t = np.round(np.cumsum(rng.exponential(1 / 200., size=20000)), 2)
    assert EventRateEstimator(t).rate == pytest.approx(200, rel=.2)
