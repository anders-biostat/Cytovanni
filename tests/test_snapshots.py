"""Snapshot tests: results on fixed synthetic inputs must match the stored ones (tests/snapshots/, marker ``snapshot``).

A failure means a result changed. If the change is intended, regenerate the case (tests/snapshots/generate.py) and
record it in CHANGELOG.md.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "snapshots"))
import cases  # noqa: E402

pytestmark = pytest.mark.snapshot


@pytest.mark.parametrize("name", list(cases.CASES))
def test_snapshot(name):
    if not os.path.exists(cases.path(name)):
        pytest.fail(f"no stored snapshot for {name}: run tests/snapshots/generate.py {name}")
    stored = dict(np.load(cases.path(name)))
    nthreads = torch.get_num_threads()
    torch.set_num_threads(1)          # the snapshots are generated with one thread
    try:
        got = cases.CASES[name]()
    finally:
        torch.set_num_threads(nthreads)
    assert set(got) == set(stored), f"{name}: keys {sorted(got)} vs stored {sorted(stored)}"
    for k in stored:
        np.testing.assert_allclose(np.asarray(got[k], dtype=float), stored[k].astype(float), rtol=cases.RTOL,
                                   atol=cases.ATOL, err_msg=f"{name}: {k}")
