"""cytovanni.rainbow.rboe: encoder of the rainbow metadata and assignment of samples to rainbow batches."""
import numpy as np
import pandas as pd
import pytest

from cytovanni.exceptions import (AutoBatchException, AutoBatchWarning, CalibrationModuleException,
                                  InconsistentBatchException, OrdinalEncoderException, OrdinalEncoderWarning,
                                  RepeatedBatchException)
from cytovanni.rainbow import RainbowBatchOrdinalEncoder

import synthetic

H = 3600.
T0 = 1.7e9


def _meta():
    """Two batches on cytometer 'cyt' (b1: two measurements, b2: one, other settings), one on 'cyt2'."""
    return pd.DataFrame({"cytometer": ["cyt", "cyt", "cyt", "cyt2"],
                         "rainbow_batch": ["b1", "b1", "b2", "c1"],
                         "rainbow_type": ["T", "T", "T", "U"],
                         "uid": ["u1", "u2", "u3", "u4"],
                         "set_hash": ["h1", "h1", "h2", "h1"],
                         "UTC": [T0, T0 + H, T0 + 5 * H, T0],
                         "date": ["d1", "d1", "d1", "d1"]})


def _sample(**uns):
    adata = synthetic.make_adata(np.zeros((2, 1)), ["FL1-A"], layer="raw")
    adata.uns.update(uns)
    return adata


def test_from_data_encodes_sorted_labels():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    assert set(rboe.oes) == set(RainbowBatchOrdinalEncoder.encode_keys)
    assert list(rboe.oes["rainbow_batch"].labels) == ["b1", "b2", "c1"]
    assert list(rboe.data_df.columns) == ["cytometer", "rainbow_batch", "rainbow_type", "uid", "set_hash", "UTC", "date"]
    assert rboe.data_df["UTC"].dtype == float
    assert rboe.oes["uid"].transform("unknown") == -1


def test_from_data_without_utc_and_date():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta().drop(columns=["UTC", "date"]))
    assert rboe.data_df["UTC"].isna().all() and rboe.data_df["date"].isna().all()


def test_batch_with_two_settings_raises():
    df = _meta()
    df.loc[1, "set_hash"] = "other"
    with pytest.raises(InconsistentBatchException):
        RainbowBatchOrdinalEncoder.from_data(df)


def test_batch_on_two_cytometers_raises():
    df = _meta()
    df.loc[3, "rainbow_batch"] = "b2"
    df.loc[3, "set_hash"] = "h2"
    with pytest.raises(RepeatedBatchException):
        RainbowBatchOrdinalEncoder.from_data(df)


def test_export_round_trip_preserves_order_and_table():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    rboe.extend(pd.DataFrame({"cytometer": ["cyt"], "rainbow_batch": ["a0"], "rainbow_type": ["T"], "uid": ["u0"],
                              "set_hash": ["h3"], "UTC": [T0 + 9 * H], "date": ["d2"]}))
    back = RainbowBatchOrdinalEncoder.from_exported(rboe.export())
    for k in rboe.encode_keys:
        assert list(back.oes[k].labels) == list(rboe.oes[k].labels)
    assert list(back.oes["rainbow_batch"].labels) == ["b1", "b2", "c1", "a0"]   # extended label appended, not sorted
    pd.testing.assert_frame_equal(back.data_df.reset_index(drop=True).astype(str),
                                  rboe.data_df.reset_index(drop=True).astype(str))
    assert back.data_df["UTC"].dtype == float


def test_extend_appends_new_ordinals():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    new = pd.DataFrame({"cytometer": ["cyt", "cyt"], "rainbow_batch": ["b2", "b3"], "rainbow_type": ["T", "V"],
                        "uid": ["u5", "u6"], "set_hash": ["h2", "h4"], "UTC": [T0, T0], "date": ["d2", "d2"]})
    new_ords = rboe.extend(new)
    assert new_ords["rainbow_batch"] == [3] and new_ords["rainbow_type"] == [2] and new_ords["uid"] == [4, 5]
    assert new_ords["cytometer"] == []
    assert rboe.oes["rainbow_batch"].transform("b1") == 0 and len(rboe.data_df) == 6


def test_extend_inconsistent_raises():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    new = _meta().iloc[[0]].assign(uid="u9", set_hash="changed")
    with pytest.raises(InconsistentBatchException):
        rboe.extend(new)


def test_check_orphan_types_R2b_G2_05():
    """R2b-G2-05: a new rainbow type is accepted if measured in a known batch or together with a known type in a new
    batch; a type only in new batches without a known type raises."""
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    in_known_batch = pd.DataFrame({"rainbow_type": ["V"], "rainbow_batch": ["b1"]})
    with_known_type = pd.DataFrame({"rainbow_type": ["V", "T"], "rainbow_batch": ["new", "new"]})
    orphan = pd.DataFrame({"rainbow_type": ["V", "V", "T"], "rainbow_batch": ["new", "new", "other"]})
    rboe.check_orphan_types(in_known_batch)
    rboe.check_orphan_types(with_known_type)
    rboe.check_orphan_types(pd.DataFrame({"rainbow_type": ["T"], "rainbow_batch": ["new"]}))   # no new type
    with pytest.raises(CalibrationModuleException, match="V"):
        rboe.check_orphan_types(orphan)


# ---------------------------------------------------------------- batch inference
def test_find_closest_batch_by_time_and_settings():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    assert rboe.find_closest_batch(_sample(date="d1", UTC=T0 + 1.2 * H, cytometer="cyt", set_hash="h1")) == "b1"
    assert rboe.find_closest_batch(_sample(date="d1", UTC=T0 + 4.5 * H, cytometer="cyt", set_hash="h2")) == "b2"
    # the settings decide before the time: closest in time is b2, but the settings are those of b1
    with pytest.warns(AutoBatchWarning):
        assert rboe.find_closest_batch(_sample(date="d1", UTC=T0 + 5 * H, cytometer="cyt", set_hash="h1")) == "b1"
    # other cytometer
    assert rboe.find_closest_batch(_sample(date="d1", UTC=T0, cytometer="cyt2", set_hash="h1")) == "c1"


def test_find_closest_batch_custom_hash_and_across_midnight():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    ad = _sample(date="d0", UTC=T0 - 2 * H, cytometer="cyt", set_hash="ignored")
    assert rboe.find_closest_batch(ad, fct_hash_settings=lambda a: "h1") == "b1"     # other date, 2 h before
    with pytest.raises(AutoBatchException):
        rboe.find_closest_batch(ad, fct_hash_settings=lambda a: "h1", max_hour_difference_across_days=1)


def test_find_closest_batch_warns_beyond_two_hours():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    with pytest.warns(AutoBatchWarning, match="hours difference"):
        rboe.find_closest_batch(_sample(date="d1", UTC=T0 + 3.5 * H, cytometer="cyt", set_hash="h1"))


@pytest.mark.parametrize("uns, match", [
    (dict(UTC=T0, cytometer="cyt", set_hash="h1"), "requires 'date'"),
    (dict(date="d1", UTC=T0, cytometer="cyt"), "set_hash"),
    (dict(date="d9", UTC=T0 + 100 * H, cytometer="cyt", set_hash="h1"), "not present"),
    (dict(date="d1", UTC=T0, cytometer="cyt", set_hash="h9"), "same settings"),
    (dict(date="d1", UTC=np.nan, cytometer="cyt", set_hash="h1"), "not valid"),
])
def test_find_closest_batch_failures_raise(uns, match):
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    with pytest.raises(AutoBatchException, match=match):
        rboe.find_closest_batch(_sample(**uns))


def test_find_closest_batch_training_without_utc_raises():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta().drop(columns=["UTC"]))
    with pytest.raises(AutoBatchException, match="no UTC"):
        rboe.find_closest_batch(_sample(date="d1", UTC=T0, cytometer="cyt", set_hash="h1"))


def test_transform_adata_batch_order_of_precedence():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    ad = _sample(rainbow_batch="b2", date="d1", UTC=T0, cytometer="cyt", set_hash="h1")
    assert rboe.transform_adata_batch(rainbow_batch="b1") == 0
    assert rboe.transform_adata_batch(adata=ad, rainbow_batch="c1") == 2       # explicit batch first
    assert rboe.transform_adata_batch(adata=ad) == 1                           # then uns['rainbow_batch']
    with pytest.warns(OrdinalEncoderWarning):
        assert rboe.transform_adata_batch(adata=ad, rainbow_batch="nope") == 1
    ad.uns["rainbow_batch"] = "default"                                        # unknown: inferred (b1 by time+hash)
    with pytest.warns(OrdinalEncoderWarning):
        assert rboe.transform_adata_batch(adata=ad, fct_hash_settings=lambda a: a.uns["set_hash"]) == 0


def test_transform_adata_batch_default_hash_from_cytoconfig():
    """Without fct_hash_settings, the batch inference uses uns['cytoconfig'].fct_hash_settings (not uns['set_hash'])."""
    from types import SimpleNamespace
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    ad = _sample(date="d1", UTC=T0 + 4.5 * H, cytometer="cyt", set_hash="h1",
                 cytoconfig=SimpleNamespace(fct_hash_settings=lambda a: "h2"))
    assert rboe.transform_adata_batch(adata=ad) == 1


def test_transform_adata_batch_errors():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    with pytest.raises(OrdinalEncoderException):
        rboe.transform_adata_batch()
    with pytest.raises(OrdinalEncoderException):
        rboe.transform_adata_batch(rainbow_batch="nope")


def test_transform_adata_unknown_and_missing_keys():
    rboe = RainbowBatchOrdinalEncoder.from_data(_meta())
    out = rboe.transform_adata(_sample(cytometer="cyt", rainbow_batch="b2", uid="zzz"))
    assert out == {"cytometer": 0, "rainbow_batch": 1, "rainbow_type": -1, "uid": -1}
