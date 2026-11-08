"""cytovanni.torch.pdsave: pandas objects as tagged numpy tuples and back."""
import numpy as np
import pandas as pd
import pytest

from cytovanni.torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict


def test_dataframe_roundtrip_loses_names():
    df = pd.DataFrame([[1., 2.], [3., 4.]], index=pd.Index(["r0", "r1"], name="rows"), columns=["a", "b"])
    ex = pdsave_expand(df)
    assert ex[0] == "_pdsave_expanded_DataFrame" and all(isinstance(e, np.ndarray) for e in ex[1:])
    back = pdsave_collect(ex)
    pd.testing.assert_frame_equal(back, df, check_names=False)
    assert back.index.name is None


def test_series_and_index_roundtrip():
    se = pd.Series([1, 2, 3], index=["x", "y", "z"])
    pd.testing.assert_series_equal(pdsave_collect(pdsave_expand(se)), se)
    idx = pd.Index(["p", "q"])
    pd.testing.assert_index_equal(pdsave_collect(pdsave_expand(idx)), idx)


def test_mixed_dtype_dataframe_comes_back_as_object():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    back = pdsave_collect(pdsave_expand(df))
    assert (back.dtypes == object).all()
    assert back.iloc[1, 0] == 2


@pytest.mark.parametrize("el", [3, "s", None, np.arange(3), (1, 2), ("a", "b"), ()])
def test_non_pandas_passes_unchanged(el):
    assert pdsave_expand(el) is el
    assert pdsave_collect(el) is el


def test_unknown_tag_raises():
    with pytest.raises(ValueError):
        pdsave_collect(("_pdsave_expanded_Panel", np.zeros(2)))


def test_corrupt_tuple_raises_instead_of_passing_through():
    """R2-E12: errors while rebuilding a tagged tuple propagate (they were swallowed, the tuple returned)."""
    with pytest.raises(ValueError):
        pdsave_collect(("_pdsave_expanded_Series", np.arange(3), np.arange(2)))


def test_dict_versions_not_recursive():
    df = pd.DataFrame([[1.]], index=["r"], columns=["c"])
    dct = {"df": df, "n": 1, "inner": {"df": df}}
    ex = pdsave_expand_dict(dct)
    assert isinstance(ex["df"], tuple) and ex["n"] == 1 and isinstance(ex["inner"]["df"], pd.DataFrame)
    back = pdsave_collect_dict(ex)
    pd.testing.assert_frame_equal(back["df"], df)
    assert dct["df"] is df  # input not modified


def test_roundtrip_through_torch_save(tmp_path):
    import torch
    df = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["c", "d"])
    torch.save(pdsave_expand_dict({"df": df}), tmp_path / "x.pt")
    back = pdsave_collect_dict(torch.load(tmp_path / "x.pt", weights_only=False))
    pd.testing.assert_frame_equal(back["df"], df)
