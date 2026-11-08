"""Stages of the tutorial pipelines on a subset of the example data, used by the end-to-end tests.

Every stage function takes its input folders and an output folder and writes its results there; the tests cache the
output folders (see ``conftest.py``). The steps follow the tutorials in docs/tutorials (same functions, settings and
gates), with plots only where they exercise plotting code, and with the data reduced to:

- fixed spectra: the rainbow beads and the stained aliquots of the last batch and of the tutorial's anchor batch
  (`FIXED_BATCHES`);
- variable spectra: the last date and the date that shares the most samples with it (`VARIABLE_DATES`), only the
  samples measured on both dates, the reference samples of both dates, and the rainbow beads of these batches and of
  the single-stain runs (all runs are needed for the spectra PCA).
"""
import os
import shutil

import numpy as np
import pandas as pd
import torch

import cytovanni
from cytovanni.gating import FluorescenceMinMaxGate, MinMaxGate, PolygonGate, gate_scatter_bead_sample
from cytovanni.io import read_rainbow_h5ad, readfcs_rainbow_sample, readfcs_sample, write_rainbow_h5ad
from cytovanni.rainbow import (RainbowCalibrator, fit_rainbow_GMM, get_rainbow_GMM_gate,
                               plot_rainbow_channel_GMM_single)

FIXED_DIR = "example_fixed-spectra"
VARIABLE_DIR = "example_variable-spectra"
FIXED_BATCHES = ["12_11_2021", "23_11_2021"]      # anchor batch of the tutorial, last batch
FIXED_ANCHOR = "12_11_2021"
VARIABLE_DATES = [20240201, 20240418]              # last date, and the date with the most shared samples (10)
VARIABLE_ANCHORS = {"batch": ["18-APR-2024"], "batch_ref": ["20240418_tcell_A"]}
FIXED_MARKERS_NORMALIZE = ['BV510', 'BV605', 'BB515', 'BUV396', 'BUV805', 'APC', 'APC-R700', 'APC-H7', 'BYG670']
VARIABLE_FIXED_INIT = "HD7_Compensation_20240418_tcell"


def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def gates_minmax(cytoconfig):
    """The two generic gates of the tutorials: scatter in [0, 4e5], fluorescence in [-1e3, 1.5 * linear range]."""
    return [MinMaxGate(["FSC-A", "FSC-H", "SSC-A", "SSC-H"], 0, 4e5, layer="raw"),
            FluorescenceMinMaxGate(cytoconfig, -1e3, 1.5, layer="raw")]


def prepare_rainbow(path, cytoconfig, npeak, rainbow_type, rainbow_batch, out, plot=False):
    """Read, gate and GMM-fit one rainbow bead measurement and write it as .h5ad (tutorial step)."""
    name = rainbow_batch + "___" + os.path.basename(path)
    ad = readfcs_rainbow_sample(path, cytoconfig, N_rainbow_peak=npeak, name=name, rainbow_type=rainbow_type,
                                rainbow_batch=rainbow_batch)
    ad = gate_scatter_bead_sample(ad, plotpath=os.path.join(out, f"gate_{rainbow_batch}.jpg") if plot else "")
    fit_rainbow_GMM(ad)
    ad = ad[get_rainbow_GMM_gate(ad, cutoff=-2, dropmaxpercent=10)].copy()
    fit_rainbow_GMM(ad)
    if plot:
        plot_rainbow_channel_GMM_single(ad, savepath=os.path.join(out, f"gmm_{rainbow_batch}.jpg"))
    write_rainbow_h5ad(ad, os.path.join(out, f"data_{ad.uns['name_uid']}.h5ad"))
    return ad


def read_rainbow_folder(folder):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".h5ad"))
    ads = [read_rainbow_h5ad(os.path.join(folder, f)) for f in files]
    ads.sort(key=lambda ad: ad.uns["UTC"])
    return ads


def fit_rainbow(prep, out):
    """`RainbowCalibrator.fit_fluorescence` (default schedule) on all prepared measurements; saves model and factors."""
    rb = RainbowCalibrator(read_rainbow_folder(prep))
    rb.fit_fluorescence(gpu=torch.cuda.is_available(), verbosity=0)
    rb.save(os.path.join(out, "RBInt.pt"))
    rb.get_calibration_factors_all().to_csv(os.path.join(out, "factors.csv"))
    pd.Series(rb.rfim_trainer.fit_losshistory).to_csv(os.path.join(out, "losshistory.csv"))


# ======================================================================== fixed spectra
def fixed_rainbow_prep(data, out):
    cfg = cytovanni.utils.CytometerConfiguration_A3small()
    raw = os.path.join(data, FIXED_DIR)
    for i, b in enumerate(FIXED_BATCHES):
        prepare_rainbow(os.path.join(raw, f"Sphero Rainbow Calibration Particles (8Peaks) {b}.fcs"), cfg, 8, "default",
                        b, out, plot=i == 0)


def fixed_rainbow_fit(prep, out):
    fit_rainbow(prep, out)


def fixed_calibrate(data, fit, out):
    """Calibrated copies of the stained aliquots (with the generic gates) and the calibrated panel spectra."""
    cfg = cytovanni.utils.CytometerConfiguration_A3small()
    raw = os.path.join(data, FIXED_DIR)
    rb = RainbowCalibrator.from_saved(os.path.join(fit, "RBInt.pt"))
    for b in FIXED_BATCHES:
        rb.make_calibrated_fcs(os.path.join(raw, f"RF_{b}.fcs"), os.path.join(out, f"RF_{b}.fcs"), cfg,
                               gates=gates_minmax(cfg))
    spectra = pd.read_table(os.path.join(raw, "panel_spectra.csv"), sep=",", index_col=0)
    rb.get_calibrated_spectra(spectra.T, FIXED_ANCHOR, spectral=False).T.to_csv(os.path.join(out, "panel_spectra_calibrated.csv"))


def fixed_read_calibrated(cal):
    cfg = cytovanni.utils.CytometerConfiguration_A3small()
    spectra = pd.read_table(os.path.join(cal, "panel_spectra_calibrated.csv"), sep=",", index_col=0)
    ads = []
    for b in FIXED_BATCHES:
        ad = readfcs_sample(os.path.join(cal, f"RF_{b}.fcs"), cfg, gates=[])
        ad.uns["date"] = f"RF_{b}"
        ad.obs["date"] = f"RF_{b}"
        ad.uns["sample"] = "Ref"
        ad.obsm["unmx"] = cytovanni.utils.apply_unmixing_lstsq(spectra, ad.to_df(layer="raw"))
        ads.append(ad)
    return ads, spectra


def fixed_ref(cal, out):
    """Scaling-factor reference model of the fixed-spectra tutorial (ScalingTransformer, Sinkhorn, 500 SGD steps)."""
    ads, spectra = fixed_read_calibrated(cal)
    scale_markers = pd.Series(False, index=spectra.index)
    scale_markers.loc[FIXED_MARKERS_NORMALIZE] = True
    moe = cytovanni.ref.MultiOrdinalEncoder.from_adatas(ads, keys_batch=["date"])
    anchor_bidx = moe.transform_dict_simple({"date": f"RF_{FIXED_ANCHOR}"})
    dataset = cytovanni.ref.SampleOverlapDataset(ads, moe, spectra.columns, key_layer="raw", key_id="sample",
                                                 key_batch_sample="date", data_isanchor=None, sN_cells=2000,
                                                 sN_persample=2, device=device(), data_on_device=True)
    transformer = cytovanni.ref.ScalingTransformer.from_data(moe, spectra, scale_stains=scale_markers,
                                                             key_scale_stains="date", arcsinh_cofactor=5000,
                                                             anchor_bidx=anchor_bidx)
    sfm = cytovanni.ref.OverlapFitModel.from_data(moe, transformer, dataset, DDIST_mode="sinkhorn",
                                                  DDIST_kwargs={"blur": .5, "reach": 1.}, DDIST_separate=False).to(device())
    sfm.fit(Niter=500, lr=2e-1, momentum=.7, optimizer="SGD")
    sfm.save(os.path.join(out, "standardisation_model.pt"))
    sfm.transformer.learned_stainscale.to_csv(os.path.join(out, "normalization_factors.csv"))
    pd.Series(sfm.losshistory if hasattr(sfm, "losshistory") else []).to_csv(os.path.join(out, "losshistory.csv"))


# ======================================================================== variable spectra
def variable_metadata(data):
    """Subsets of the tutorial's metadata tables: (samples, references, rainbow, single-stain runs)."""
    raw = os.path.join(data, VARIABLE_DIR)
    smp = pd.read_csv(os.path.join(raw, "raw_fcs", "meta_SamplesOverlap.csv"), index_col=0)
    ref = pd.read_csv(os.path.join(raw, "raw_fcs", "meta_ReferenceOverlap.csv"), index_col=0)
    rbw = pd.read_csv(os.path.join(raw, "raw_fcs", "meta_Rainbow.csv"), index_col=0)
    runs = pd.read_csv(os.path.join(raw, "single_stain", "meta_run.csv"), index_col=0)
    smp = smp[smp["m_date"].isin(VARIABLE_DATES)]
    shared = set.intersection(*[set(smp.loc[smp["m_date"] == d, "ovlp_uid"]) for d in VARIABLE_DATES])
    smp = smp[smp["ovlp_uid"].isin(shared)]
    ref = ref[ref["m_date"].isin(VARIABLE_DATES)]
    rbw = rbw[rbw["batch_rb"].isin(set(smp["batch_rb"]) | set(ref["batch_rb"]) | set(runs["rainbow_batch"]))].copy()
    rbw["type_rb"] = rbw["type_rb"].astype(str)
    return smp, ref, rbw, runs


def variable_rainbow_prep(data, out):
    cfg = cytovanni.utils.CytometerConfiguration_A3()
    raw = os.path.join(data, VARIABLE_DIR, "raw_fcs")
    smp, ref, rbw, runs = variable_metadata(data)
    for i, (_, row) in enumerate(rbw.iterrows()):
        prepare_rainbow(os.path.join(raw, row["filepath"]), cfg, row["type_rb_Npeak"], row["type_rb"], row["batch_rb"],
                        out, plot=i == 0)


def variable_rainbow_fit(prep, out):
    fit_rainbow(prep, out)


def variable_calibrate(data, fit, out):
    """Calibrated copies of the selected sample and reference files (generic gates), calibrated single-stain spectra
    of every run, and the metadata tables of the subset."""
    cfg = cytovanni.utils.CytometerConfiguration_A3()
    raw = os.path.join(data, VARIABLE_DIR)
    rb = RainbowCalibrator.from_saved(os.path.join(fit, "RBInt.pt"))
    smp, ref, rbw, runs = variable_metadata(data)
    for meta in [smp, ref]:
        for fp in meta["filepath"]:
            dst = os.path.join(out, "calibrated_fcs", fp)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            rb.make_calibrated_fcs(os.path.join(raw, "raw_fcs", fp), dst, cfg, gates=gates_minmax(cfg))
    smp.to_csv(os.path.join(out, "calibrated_fcs", "meta_SamplesOverlap.csv"))
    ref.to_csv(os.path.join(out, "calibrated_fcs", "meta_ReferenceOverlap.csv"))
    for _, row in runs.iterrows():
        src = os.path.join(raw, "single_stain", row["run"])
        dst = os.path.join(out, "single_stain", row["run"])
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(src, "panel_meta.csv"), dst)
        spectra = pd.read_table(os.path.join(src, "spectra.csv"), sep=",", index_col=0)
        rb.get_calibrated_spectra(spectra, row["rainbow_batch"], spectral=False).to_csv(os.path.join(dst, "spectra_calibrated.csv"))
    runs.to_csv(os.path.join(out, "single_stain", "meta_run.csv"))


def variable_dye(data, cal, out):
    """Spillover collection of the calibrated spectra, PCA per dye, and the T-cell panel configuration."""
    raw = os.path.join(data, VARIABLE_DIR)
    runs = pd.read_csv(os.path.join(cal, "single_stain", "meta_run.csv"), index_col=0)["run"]
    key_dye_channel = pd.read_table(os.path.join(raw, "single_stain", "meta_dye_channel.csv"), sep=",", index_col=0)["channel"]
    sps = pd.Series([cytovanni.spill.SpilloverMeasurement.from_csv(
        os.path.join(cal, "single_stain", r, "spectra_calibrated.csv"), os.path.join(cal, "single_stain", r, "panel_meta.csv"),
        is_spectral=False, clip_negative=True) for r in runs], index=runs.to_numpy())
    spc = cytovanni.spill.SpilloverCollection(sps, register_on="dye")
    spc.normalize_spectra(key_dye_channel)
    spc.add_PCA_fit("spectra")
    pc = cytovanni.spill.PanelConfiguration.from_csv(os.path.join(raw, "single_stain", "panel_HD7_TCell.csv"), "HD7_TCell")
    pc.collect_PCA_data(spc)
    pc.save(os.path.join(out, "panel_HD7_TCell.pt"))


VARIABLE_GATES = dict(
    cells=(("FSC-A", "SSC-A"), [[6e4, 1.5e4], [8e4, 1.3e4], [1.35e5, 5e4], [1.35e5, .9e5], [.9e5, .9e5], [.7e5, 4.5e4]]),
    nd_SSC=(("SSC-A", "SSC-H"), [[1.5e4, .5e4], [1e4, 1.5e4], [1e5, 1e5], [1e5, 7e4]]),
    nd_FSC=(("FSC-A", "FSC-H"), [[6e4, 4e4], [5.5e4, 6e4], [1.4e5, 1.3e5], [1.4e5, 10e4]]),
)


def variable_load_samples(cal):
    """Read the calibrated sample and reference files with the tutorial's cell and doublet gates and batch keys."""
    cfg = cytovanni.utils.CytometerConfiguration_A3()
    gates = [PolygonGate(x, y, np.asarray(p), name=k, layer="raw") for k, ((x, y), p) in VARIABLE_GATES.items()]
    out = []
    for meta_name in ["meta_ReferenceOverlap.csv", "meta_SamplesOverlap.csv"]:
        meta = pd.read_csv(os.path.join(cal, "calibrated_fcs", meta_name), index_col=0)
        for _, row in meta.iterrows():
            ad = readfcs_sample(os.path.join(cal, "calibrated_fcs", row["filepath"]), cfg, metadct=row.to_dict())
            for g in gates:
                ad = ad[g.apply(ad)]
            ad = ad.copy()
            ad.uns["batch"] = ad.uns["date"] + ("_late" if ad.uns["batch_ref"] == "20240201_tcell_E" else "")
            ad.uns["order_batch_ref"] = ord(ad.uns["batch_ref"][-1]) - 65
            out.append(ad)
    return out


def variable_model(cal, dye, out, Niter=1500):
    """Panel-embedding reference model of the variable-spectra tutorial (SpectralFitTransformer, Sinkhorn)."""
    adatas = variable_load_samples(cal)
    train = [ad for ad in adatas if ad.uns["is_train"]]
    pc = cytovanni.spill.PanelConfiguration.from_saved(os.path.join(dye, "panel_HD7_TCell.pt"))
    scale_stains = pd.Series(True, index=pc.stains)
    scale_stains.loc[["BUV661", "BV605", "BB700", "PE-CF594", "PE-Cy7 (BD)", "eflour"]] = False
    key_comp_batch = pd.Series("batch", index=pc.pca_components.index.to_numpy(), dtype=str)
    key_comp_batch.loc[pc.pca_components_stain == "BV786"] = "batch_ref"
    moe = cytovanni.ref.MultiOrdinalEncoder.from_adatas(train, keys_batch=["batch", "batch_ref"],
                                                        nested={"batch_ref": ["batch", "order_batch_ref"]})
    anchor_bidx = moe.transform_dict_simple(VARIABLE_ANCHORS)
    dataset = cytovanni.ref.SampleOverlapDataset(train, moe, pc.channels, keys_sample_covariate=["ovlp_uid"],
                                                 key_id="ovlp_uid", key_layer="raw", key_batch_sample="batch",
                                                 data_isanchor=cytovanni.ref.adatas_process_anchors(train, VARIABLE_ANCHORS),
                                                 sN_cells=1500, sN_persample=2, device=device(), data_on_device=True)
    spectra = pd.read_table(os.path.join(cal, "single_stain", VARIABLE_FIXED_INIT, "spectra_calibrated.csv"), sep=",",
                            index_col=0).T
    embedding = pc.get_spectra_embedding(spectra, ensure_complete=True)
    pem = cytovanni.ref.PanelEmbeddingModule.from_data(pc, moe, key_comp_batch, batch_order=["batch", "batch_ref"],
                                                       anchor_bidx=anchor_bidx, init_emb=embedding,
                                                       max_abs_embedding=6).to(device())
    transformer = cytovanni.ref.SpectralFitTransformer.from_data(moe, pem, scale_stains=scale_stains,
                                                                 key_scale_stains="batch", arcsinh_cofactor=5000,
                                                                 L1weight_embedding=2e-7, L1weight_stainscale=1e-7,
                                                                 L1weight_channelscale=1e-7, anchor_bidx=anchor_bidx)
    sfm = cytovanni.ref.OverlapFitModel.from_data(moe, transformer, dataset, DDIST_mode="sinkhorn",
                                                  DDIST_kwargs={"blur": .5, "reach": 1.}).to(device())
    sfm.save(os.path.join(out, "model_init.pt"))
    sfm.fit(Niter=Niter, lr=2e-2)
    sfm.save(os.path.join(out, "model.pt"))
    sfm.transformer.learned_stainscale.to_csv(os.path.join(out, "stainscale.csv"))


def aliquot_mismatch(adatas, key, cofactor=1500, quantiles=(.5, .9)):
    """Mean absolute difference of arcsinh quantiles between aliquots of the same sample (``ovlp_uid``) measured in
    different batches, per marker (rows: marker; one value per quantile)."""
    rows = {}
    by_uid = {}
    for ad in adatas:
        by_uid.setdefault(ad.uns["ovlp_uid"], []).append(ad)
    for uid, ads in by_uid.items():
        if len({ad.uns["batch"] for ad in ads}) < 2:
            continue
        q = [np.arcsinh(ad.obsm[key] / cofactor).quantile(list(quantiles)) for ad in ads]
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                if ads[i].uns["batch"] != ads[j].uns["batch"]:
                    rows[(uid, i, j)] = (q[i] - q[j]).abs().mean(0)
    return pd.DataFrame(rows).T.mean(0)
