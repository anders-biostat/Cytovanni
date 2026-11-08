"""End-to-end tests on the example data (marker ``data``; not part of the default run).

The tutorial pipelines run on a subset of the example data: the last batch and one other batch, overlapping
(matching) samples only. The pipelines are split into stages (rainbow calibration, single-stain spectra, reference
model, ...); every stage writes its results to a cache folder, so later stages can run on their own when the cache
exists, and fast stages can run without the slow ones (e.g. ``pytest -m "data and not slow" tests/e2e``).

Environment:
- ``CYTOVANNI_TEST_DATA``: folder with the example data sets (e.g. ``example_fixed-spectra``); the tests are skipped
  if it is not set or missing.
- ``CYTOVANNI_TEST_CACHE``: folder for the stage outputs (default: a pytest temporary folder, i.e. no reuse between
  runs). A stage counts as done when its folder holds a ``DONE`` file; delete the folder to rerun the stage.
- ``CYTOVANNI_TEST_REFERENCE``: optional folder with the outputs of the full tutorial runs (the docs/tutorials folder
  after running the notebooks); the comparisons with the full runs are skipped without it.

Stages that train a model are marked ``slow``. A stage that is not cached is computed when a test needs it, unless
slow tests are deselected (``-m "... not slow"``): then the tests that need it are skipped.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import pipeline  # noqa: E402


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


@pytest.fixture(scope="session")
def reference_outputs():
    """Folder with the outputs of the full tutorial runs (``CYTOVANNI_TEST_REFERENCE``), or None."""
    root = os.environ.get("CYTOVANNI_TEST_REFERENCE")
    return root if root and os.path.isdir(root) else None


def run_stage(request, cache, name, fn, *args, slow=False):
    """Output folder of stage `name`: from the cache if done, else computed with ``fn(*args, folder)``."""
    folder = os.path.join(cache, name)
    done = os.path.join(folder, "DONE")
    if os.path.exists(done):
        return folder
    if slow and "not slow" in (request.config.option.markexpr or ""):
        pytest.skip(f"stage {name} is not cached and slow stages are deselected")
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    fn(*args, folder)
    with open(done, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {time.time() - t0:.0f} s\n")
    return folder
