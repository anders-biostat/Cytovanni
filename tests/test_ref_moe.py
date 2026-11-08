"""Tests of cytovanni.ref.moe: MultiOrdinalEncoder (batch labels -> batch indices, nested imputation)."""
import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.ref import MultiOrdinalEncoder


@pytest.fixture
def meta_nested():
    """Plates nested in days, with an order (plate number) within the day."""
    return pd.DataFrame({"plate": ["p1", "p2", "p3", "p4"], "day": ["d1", "d1", "d2", "d2"], "order": [0, 20, 0, 5]})


@pytest.fixture
def moe_nested(meta_nested):
    return MultiOrdinalEncoder.from_data(meta_nested, ["plate", "day"], nested={"plate": ["day", "order"]})


def test_from_data_sorts_labels_as_str_and_counts_batches():
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"b": ["z", "a", 3, "a"]}), ["b"])
    assert moe.labels["b"].tolist() == ["3", "a", "z"]
    assert moe.Nbatch == {"b": 3}
    assert len(moe["b"]) == 3
    assert list(moe.keys()) == ["b"]
    assert dict(moe.items())["b"] is moe["b"]


def test_necessary_keys_include_nested_major_and_order(moe_nested):
    assert set(moe_nested.necessary_keys) == {"plate", "day", "order"}


def test_from_adatas_reads_uns():
    ads = [synthetic.cell_adata(b, n=10, seed=i) for i, b in enumerate(synthetic.BATCHES)]
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    assert moe.labels["batch"].tolist() == synthetic.BATCHES
    out = moe.transform_ad(ads[1])
    assert out["batch"].dtype == torch.long and out["batch"].tolist() == [1]


def test_transform_df_returns_long_tensors_or_numpy():
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"b": ["x", "y"]}), ["b"])
    out = moe.transform_df(pd.DataFrame({"b": ["y", "x", "y"]}))
    assert isinstance(out["b"], torch.Tensor) and out["b"].dtype == torch.long
    assert out["b"].tolist() == [1, 0, 1]
    out_np = moe.transform_df(pd.DataFrame({"b": ["y"]}), astorch=False)
    assert isinstance(out_np["b"], np.ndarray)


def test_transform_df_unknown_label_of_non_nested_key_raises():
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"b": ["x", "y"]}), ["b"])
    with pytest.raises(ValueError):
        moe.transform_df(pd.DataFrame({"b": ["new"]}))


def test_nested_imputation_nearest_order_within_major(moe_nested):
    q = pd.DataFrame({"plate": ["pX", "pY"], "day": ["d1", "d2"], "order": [12, 4]})
    out = moe_nested.transform_df(q)
    assert moe_nested.labels["plate"][out["plate"].numpy()].tolist() == ["p2", "p4"]
    assert out["plate_imputed"].tolist() == [True, True]
    assert out["plate_imputed"].dtype == torch.bool


def test_nested_imputation_tie_goes_to_lower_order(moe_nested):
    out = moe_nested.transform_df(pd.DataFrame({"plate": ["pX"], "day": ["d1"], "order": [10]}))
    assert moe_nested.labels["plate"][out["plate"].numpy()].tolist() == ["p1"]


def test_nested_imputation_keeps_known_rows():
    """R2-A4: rows with a known nested label keep it when other rows of the same call are imputed."""
    meta = pd.DataFrame({"day": ["d1", "d2", "d3"], "month": ["m1"] * 3, "time": [0, 10, 20]})
    moe = MultiOrdinalEncoder.from_data(meta, ["day", "month"], nested={"day": ["month", "time"]})
    out = moe.transform_df(pd.DataFrame({"day": ["d1", "dX", "d3"], "month": ["m1"] * 3, "time": [9, 11, 0]}))
    assert moe.labels["day"][out["day"].numpy()].tolist() == ["d1", "d2", "d3"]
    assert out["day_imputed"].tolist() == [False, True, False]


def test_nested_imputation_unknown_major_without_reference_raises():
    """R2b-G2-04: an unknown nested label whose major batch has no known nested batch raises, naming the value."""
    ref = pd.DataFrame({"minor": ["a", "b"], "major": ["M1", "M2"], "order": [0, 0]})
    moe = MultiOrdinalEncoder.from_data(ref, ["minor"], nested={"minor": ["major", "order"]})
    with pytest.raises(ValueError, match="UNSEEN"):
        moe.transform_df(pd.DataFrame({"minor": ["new", "a"], "major": ["UNSEEN", "M1"], "order": [1, 0]}))
    out = moe.transform_df(pd.DataFrame({"minor": ["new", "a", "b"], "major": ["M2", "M1", "M2"], "order": [5, 0, 0]}))
    assert moe["minor"].labels[out["minor"].numpy()].tolist() == ["b", "a", "b"]
    assert out["minor_imputed"].tolist() == [True, False, False]


def test_transform_dict_is_one_row_transform_df(moe_nested):
    out = moe_nested.transform_dict({"plate": "p3", "day": "d2", "order": 0})
    assert out["plate"].tolist() == [2] and out["day"].tolist() == [1] and out["plate_imputed"].tolist() == [False]


def test_extract_ad_and_transform_ad_need_all_necessary_keys(moe_nested):
    ad = synthetic.make_adata(np.zeros((2, 3)), synthetic.CHANNELS_FLUORESCENCE, plate="p2", day="d1", order=20)
    assert moe_nested.extract_ad(ad) == {"plate": "p2", "day": "d1", "order": 20}
    assert moe_nested.transform_ad(ad)["plate"].tolist() == [1]
    del ad.uns["order"]
    with pytest.raises(KeyError):
        moe_nested.transform_ad(ad)


def test_transform_dict_simple_lists_and_single_labels(moe_nested):
    out = moe_nested.transform_dict_simple({"plate": ["p4", "p1"], "day": "d2", "notakey": ["x"]})
    assert "notakey" not in out
    assert np.asarray(out["plate"]).tolist() == [3, 0]
    assert int(np.asarray(out["day"]).ravel()[0]) == 1


def test_export_from_exported_roundtrip(moe_nested):
    moe2 = MultiOrdinalEncoder.from_exported(moe_nested.export())
    assert moe2.keys_batch == moe_nested.keys_batch
    assert moe2.Nbatch == moe_nested.Nbatch
    assert set(moe2.necessary_keys) == set(moe_nested.necessary_keys)
    q = pd.DataFrame({"plate": ["pX", "p3"], "day": ["d1", "d2"], "order": [12, 0]})
    a, b = moe_nested.transform_df(q), moe2.transform_df(q)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_extend_keeps_old_indices_and_returns_new(moe_nested):
    new = moe_nested.extend(pd.DataFrame({"plate": ["p0", "p2"], "day": ["d3", "d1"], "order": [0, 20]}))
    assert new["plate"] == [4] and new["day"] == [2]
    assert moe_nested.labels["plate"].tolist() == ["p1", "p2", "p3", "p4", "p0"]
    assert moe_nested.Nbatch == {"plate": 5, "day": 3}
    # the new nested batch is now a reference for imputation in its major batch
    out = moe_nested.transform_df(pd.DataFrame({"plate": ["pZ"], "day": ["d3"], "order": [3]}))
    assert moe_nested.labels["plate"][out["plate"].numpy()].tolist() == ["p0"]


def test_extend_without_new_labels_returns_empty_lists():
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"b": ["x", "y"]}), ["b"])
    assert moe.extend(pd.DataFrame({"b": ["y"]})) == {"b": []}
