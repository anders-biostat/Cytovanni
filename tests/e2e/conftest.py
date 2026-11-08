"""End-to-end tests on the example data (marker ``data``; not part of the default run).

The tutorial pipelines run on a subset of the example data: the last batch and one other batch, overlapping
(matching) samples only. The pipelines are split into stages (rainbow calibration, single-stain spectra, reference
model, ...); every stage writes its results to a cache folder, so later stages can run on their own when the cache
exists, and fast stages can run without the slow ones (e.g. ``pytest -m "data and not slow" tests/e2e``).

Environment:
- ``CYTOVANNI_TEST_DATA``: folder with the example data sets (e.g. ``example_fixed-spectra``); the tests are skipped
  if it is not set or missing.
- ``CYTOVANNI_TEST_CACHE``: folder for the stage outputs (default: a pytest temporary folder, i.e. no reuse between
  runs).
"""
import os
import pytest


def pytest_collection_modifyitems(config, items):
    """Mark every test in this folder as ``data``."""
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.data)


@pytest.fixture(scope="session")
def example_data():
    """Root folder of the example data (``CYTOVANNI_TEST_DATA``); skips the test if it is not available."""
    root = os.environ.get("CYTOVANNI_TEST_DATA")
    if not root or not os.path.isdir(root):
        pytest.skip("example data not available (set CYTOVANNI_TEST_DATA)")
    return root


@pytest.fixture(scope="session")
def stage_cache(tmp_path_factory):
    """Folder where the stages store their outputs (``CYTOVANNI_TEST_CACHE`` or a temporary folder)."""
    root = os.environ.get("CYTOVANNI_TEST_CACHE")
    if root:
        os.makedirs(root, exist_ok=True)
        return root
    return str(tmp_path_factory.mktemp("e2e_cache"))
