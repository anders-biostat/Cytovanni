"""Shared fixtures for the Cytovanni tests: seeds, devices, the simulated cytometer and its files.

Markers (registered in pyproject.toml): ``slow``, ``gpu``, ``keops``, ``data``, ``snapshot``. The default run
(``pytest``) excludes ``data`` and ``keops``; GPU tests are skipped without CUDA.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))  # makes `synthetic` importable from the test modules
import synthetic  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _seed_and_close_figures():
    """Fixed seeds for numpy's global state and torch in every test; close figures afterwards."""
    np.random.seed(0)
    torch.manual_seed(0)
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def rng():
    """A fresh numpy Generator with a fixed seed."""
    return np.random.default_rng(0)


CUDA = torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not CUDA, reason="no CUDA")])]


@pytest.fixture(params=DEVICES)
def device(request):
    """Runs a test on the CPU and, if available, on CUDA (``gpu`` marker)."""
    return request.param


@pytest.fixture(scope="session")
def config():
    """CytometerConfiguration of the simulated cytometer (see tests/synthetic.py)."""
    return synthetic.make_config()


@pytest.fixture(scope="session")
def rainbow_files(tmp_path_factory):
    """Rainbow bead .fcs files: two measurements per batch, dict batch -> list of paths."""
    folder = tmp_path_factory.mktemp("rainbow")
    return {b: [synthetic.write_rainbow_fcs(str(folder / f"rainbow_{b}_{k}.fcs"), b, btim=f"0{8+k}:00:00", seed=10 * i + k)
                for k in range(2)] for i, b in enumerate(synthetic.BATCHES)}


@pytest.fixture(scope="session")
def cell_files(tmp_path_factory):
    """Stained-cell .fcs files, one per batch with the same abundances (an overlapping sample), dict batch -> path."""
    folder = tmp_path_factory.mktemp("cells")
    ab = synthetic.cell_abundances(4000, np.random.default_rng(1))
    return {b: synthetic.write_cell_fcs(str(folder / f"cells_{b}.fcs"), b, seed=20 + i, abundances=ab)
            for i, b in enumerate(synthetic.BATCHES)}


@pytest.fixture(scope="session")
def single_stain_folder(tmp_path_factory):
    """Folder with single-stain controls of every stain and an unstained control (batch A)."""
    return synthetic.write_single_stains(str(tmp_path_factory.mktemp("single_stain")))


@pytest.fixture
def no_warnings():
    """Turn warnings into errors inside the test (use for code paths that must run silently)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
