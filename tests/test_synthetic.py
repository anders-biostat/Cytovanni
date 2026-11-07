"""Checks of the test infrastructure itself: the synthetic data match their ground truth and read back correctly."""
import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.io import readfcs_sample, readfcs_rainbow_sample, read_folder_timestamps


def test_config_channels(config):
    assert config.channels_fluorescence == synthetic.CHANNELS_FLUORESCENCE
    assert set(synthetic.CHANNELS_SCATTER) <= set(config.channels_scatter)
    assert config.channel_laser_key["FL3-A"] == "Red"


def test_spectra_normalised():
    s = synthetic.SPECTRA
    assert np.allclose(np.diag(s.to_numpy()), 1.)
    assert list(s.index) == synthetic.STAINS and list(s.columns) == synthetic.CHANNELS_FLUORESCENCE


def test_rainbow_events_follow_gains():
    for b in synthetic.BATCHES:
        df, label = synthetic.rainbow_events(b, n=6000, seed=1)
        brightest = df.loc[label == len(synthetic.PEAKS) - 1, synthetic.CHANNELS_FLUORESCENCE].median()
        expected = synthetic.PEAKS[-1] * synthetic.GAINS.loc[b]
        assert np.allclose(brightest.to_numpy(), expected.to_numpy(), rtol=.02)


def test_rainbow_file_reads_back(rainbow_files, config):
    path = rainbow_files["B"][0]
    adata = readfcs_rainbow_sample(path, config, len(synthetic.PEAKS), name="rb", rainbow_batch="B")
    assert adata.shape[1] == len(config.channels_scatter) + len(config.channels_fluorescence)
    assert adata.uns["date"] == synthetic.DATES["B"]
    assert adata.uns["N_rainbow_peak"] == len(synthetic.PEAKS)
    raw = adata.to_df(layer="raw")
    assert raw["FL1-A"].max() > synthetic.PEAKS[-1]


def test_cell_files_share_abundances(cell_files, config):
    reads = {b: readfcs_sample(p, config) for b, p in cell_files.items()}
    a, b = reads["A"].to_df(layer="raw"), reads["B"].to_df(layer="raw")
    assert a.shape == b.shape
    # same abundances, different gains/stain scales: the medians differ, but are positively related channel by channel
    assert not np.allclose(a[synthetic.CHANNELS_FLUORESCENCE].median(), b[synthetic.CHANNELS_FLUORESCENCE].median())


def test_cell_files_chronological(cell_files):
    import os
    tps = read_folder_timestamps(os.path.dirname(cell_files["A"]))
    assert [p.split("_")[-1] for p in tps.index] == ["A.fcs", "B.fcs", "C.fcs"]


def test_single_stain_folder(single_stain_folder):
    import os
    files = sorted(os.listdir(single_stain_folder))
    assert len(files) == len(synthetic.STAINS) + 1
    assert any(f.startswith("Unstained") for f in files)


def test_cell_adata_scale():
    adata = synthetic.cell_adata("C", n=500, seed=3)
    assert adata.layers["calibrated"].shape == (500, 3)
    assert adata.uns["batch"] == "C" and adata.uns["uid"] == "C"


def test_device_fixture(device):
    x = torch.ones(3, device=device)
    assert x.device.type == device
