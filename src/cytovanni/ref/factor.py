"""Per-batch scaling factors for batch integration without a full reference model.

Every function here returns one multiplicative factor per batch (and, for `fit_factors_Sinkhorn`, per marker):
multiply the batch's data (on the linear scale the factors were fitted on, raw or calibrated) by its factor to
bring the batches onto a common scale. The Sinkhorn fits match whole distributions on the arcsinh scale; mean and
percentile matching are quick closed-form alternatives (also used as initialisation for the Sinkhorn fits).
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..rainbow.rbim_data import Sampler
from ..torch import GeomSampleLoss, ExtendableParameter
from .model import _sum_loss_separate_markers
import warnings

def _triu_indices(N, Nmax=None, exclude_internal=[]):
    """Index pairs (i < j) of `N` items, shape (2, pairs); a random subset of at most `Nmax` pairs if given."""
    ind_triu = torch.triu_indices(N,N, offset=1)
    ind_triu = ind_triu[:,~torch.all(torch.isin(ind_triu, torch.tensor(exclude_internal)), axis=0)]
    if Nmax is None:
        return ind_triu
    else:
        return ind_triu[:,torch.randperm(ind_triu.shape[1])][:,:Nmax]

def _check_nonempty(sizes):
    """Raise ValueError if any entry of `sizes` (dict of name to number of events) is 0."""
    empty = [k for k, n in sizes.items() if n==0]
    if len(empty)>0:
        raise ValueError(f"No events in {empty}, every batch needs data for the factor fit!")

def fit_factor_Sinkhorn(datadct, arcsinh_cofactor=5000, Nsamp=2000, Niter=150, lr=5e-1, momentum=.8, device="cuda:0", verbose=True, aggregate=True, Ncomp=10, init="null", anchor=None, dummy=False, name=""):
    """Fit one scaling factor per batch for one marker by matching distributions with a Sinkhorn loss.

    Finds factors such that the arcsinh-transformed, rescaled distributions of all batches overlap as well as
    possible. Unlike mean or percentile matching, this uses the whole distribution, so it is robust to slightly
    changed populations, but it assumes the batches have approximately the same composition (e.g. the same reference
    sample).

    Parameters
    ----------
    datadct : dict of str to numpy.ndarray
        Events of one marker per batch: 1-D array of intensities on a linear scale (raw or calibrated, unmixed;
        not arcsinh-transformed), e.g. ``{b: df[marker].to_numpy() for b, df in unmx_by_batch.items()}``.
        The dict order sets the order of the result.
    arcsinh_cofactor : float, default 5000
        Cofactor (raw intensity units) of the transform used inside the loss: the loss compares
        ``arcsinh(x * factor / arcsinh_cofactor)`` between batches. The factors are applied to the linear data.
    Nsamp : int, default 2000
        Events drawn per batch in every step (a new random subsample each step; batches with fewer events contribute
        all of them). Also the size of the pooled comparison sample when `aggregate`.
    Niter : int, default 150
        Number of SGD steps (no convergence check: inspect the plots shown with `verbose`).
    lr : float, default 0.5
        SGD learning rate (on the log factors).
    momentum : float, default 0.8
        SGD momentum.
    device : str, default "cuda:0"
        Torch device for the fit, e.g. ``"cpu"``, ``"cuda:0"``. On the CPU the fit is much slower (minutes per fit).
    verbose : bool, default True
        Show a tqdm progress bar and, after the fit, plot the loss and the log factors per step (``plt.show()``).
    aggregate : bool, default True
        If True, compare every batch to a pooled sample (`Nsamp` events drawn from the rescaled subsamples of all
        batches, including the batch itself); the loss is the mean over batches. If False, compare pairs of batches
        directly (see `Ncomp`).
    Ncomp : int, default 10
        Only if not `aggregate`: number of random batch pairs compared per step (all pairs if there are fewer); the
        loss is the mean over these pairs.
    init : {'null', 'mean'}, default 'null'
        Initial factors: 'null' starts all factors at 1, 'mean' at the result of `fit_factor_MeanMatching`.
    anchor : str, optional
        Batch whose factor is set to 1 after the fit (all factors are divided by its factor). Default: no anchor, the
        factors are normalised to a geometric mean of 1 over all batches.
    dummy : bool, default False
        Skip the fit (and all checks) and return 1 for every batch.
    name : str, default ""
        Label of the progress bar.

    Returns
    -------
    pandas.Series
        Factor per batch (index: the keys of `datadct`, in dict order). Multiply the linear data of each batch by its
        factor to standardise it.

    Raises
    ------
    ValueError
        If `anchor` is not a key of `datadct`, or a batch has no events.
    NotImplementedError
        If `init` is not 'null' or 'mean'.

    See Also
    --------
    fit_factors_Sinkhorn : several markers at once (joint or per-marker loss).
    fit_factor_Sinkhorn_grouped : several samples per batch.
    fit_factor_MeanMatching, fit_factor_PercentileMatching : closed-form alternatives.

    Notes
    -----
    The fitted parameters are the log factors, constrained to mean 0 over batches (so the factors have geometric mean
    1); this fixes the otherwise free common scale and keeps the fit stable. `anchor` only rescales the result.

    The loss is the debiased Sinkhorn divergence of ``geomloss.SamplesLoss`` (via `cytovanni.torch.GeomSampleLoss`)
    with p=2 (squared Euclidean cost), blur 0.5 (entropic regularisation eps = blur^2), reach 1 (unbalanced transport,
    marginal penalty rho = reach^2) and scaling 0.5, uniform weights. Blur and reach are fixed here; together with the
    default cofactor of 5000 they were tuned for a 23-colour panel and may need different values for other data
    (not exposed as arguments).

    Examples
    --------
    >>> factors = fit_factor_Sinkhorn({b: df["CD3"].to_numpy() for b, df in unmx_by_batch.items()},
    ...                               Niter=200, init="mean", device="cuda:0")
    >>> standardised = {b: df["CD3"] * factors[b] for b, df in unmx_by_batch.items()}
    """
    # prepare data
    keys = np.asarray(list(datadct.keys()))
    Nbatch = len(keys)

    if dummy:
        return pd.Series(1., index=keys)
    
    if anchor is not None and not anchor in keys:
        raise ValueError(f"Anchor needs to be one of the data dict keys, '{anchor}' is not available! {list(keys)}")
    _check_nonempty({k:len(v) for k, v in datadct.items()})
    
    data_torch_split = [torch.tensor(v, dtype=torch.float32) for v in datadct.values()]
    data_torch = torch.cat(data_torch_split)[:,None].to(device)
    
    borders = np.cumsum([0]+[d.shape[0] for d in data_torch_split])
    samplerinds = [torch.arange(borders[i],borders[i+1]) for i in range(len(borders)-1)]
    sampler = Sampler(samplerinds, Nsamp, device=device)
    
    # prepare fit
    fct_loss = GeomSampleLoss("sinkhorn", blur=.5, reach=1.)
    fct_ash = lambda x: torch.arcsinh(x / arcsinh_cofactor)
    if init=="null":
        logfactor_init = torch.zeros((Nbatch,), dtype=torch.float32)
    elif init=="mean":
        logfactor_init = torch.tensor(fit_factor_MeanMatching(datadct).loc[keys].to_numpy(), dtype=torch.float32).log()
    else:
        raise NotImplementedError(f"Initialisation '{init}' not available!")

    param_logfactor = ExtendableParameter((Nbatch,),0,0, allow_grad=True, axis_norm=0)
    param_logfactor.set_trainable_part(logfactor_init)
    param_logfactor.to(device)
    
    def _get_loss():
        inds, group = sampler.sample(split=True)
        triu_comp = _triu_indices(Nbatch, Ncomp)
        factors = param_logfactor().exp()
        datas_scaled = [factors[i] * data_torch[inds[i]] for i in range(Nbatch)]

        if aggregate:
            loss = 0.
            data_comp = torch.cat(datas_scaled, 0)
            data_comp = data_comp[torch.randperm(data_comp.shape[0])][:Nsamp]
            for i in range(Nbatch):
                loss += fct_loss(fct_ash(datas_scaled[i]), fct_ash(data_comp))
            loss /= Nbatch
        else:
            loss = 0.
            for i, j in zip(triu_comp[0].numpy(), triu_comp[1].numpy()):
                loss += fct_loss(fct_ash(datas_scaled[i]), fct_ash(datas_scaled[j]))
            loss /= triu_comp.shape[1]
    
        return loss
    
    # fit
    optimizer = torch.optim.SGD(param_logfactor.parameters(), lr=lr, momentum=momentum)
    losshistory = []
    paramhistory = []

    with tqdm(range(Niter), desc=name, disable=not verbose) as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = _get_loss()
            loss.backward()
            losshistory.append(loss.item())
            paramhistory.append(param_logfactor().detach().cpu().numpy())
            optimizer.step()
    
    if verbose:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(losshistory)
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("OT Loss")
        ax[1].plot(np.asarray(paramhistory))
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("Scale Factor [log]")
        plt.show()

    factors = pd.Series(param_logfactor().exp().detach().cpu().numpy(), index=keys)
    # normalize here, fit more stable if only mean is fixed
    if anchor is not None:
        factors = factors / factors.loc[anchor]
        
    return factors

def fit_factor_Sinkhorn_grouped(datadct, arcsinh_cofactor=5000, Nsamp=2000, Niter=150, lr=5e-1, momentum=.8, device="cuda:0", verbose=True, init="null", anchor=None, dummy=False, name=""):
    """Fit one scaling factor per batch for one marker, with several samples per batch.

    Like `fit_factor_Sinkhorn` with ``aggregate=True``, but each batch holds several samples and shares one factor
    across them. Each (batch, sample) entry is compared only to the pooled same sample across batches, so differences
    in composition between samples do not enter the fit.

    Parameters
    ----------
    datadct : dict of str to dict of str to numpy.ndarray
        ``{batch: {sample: events}}``: events of one marker as 1-D array-like of intensities on a linear scale (raw or
        calibrated, unmixed; not arcsinh-transformed). Samples are matched across batches by name. The outer dict
        order sets the order of the result.
    arcsinh_cofactor : float, default 5000
        Cofactor (raw intensity units) of the transform used inside the loss: the loss compares
        ``arcsinh(x * factor / arcsinh_cofactor)`` between batches. The factors are applied to the linear data.
    Nsamp : int, default 2000
        Events drawn per (batch, sample) entry in every step (a new random subsample each step; entries with fewer
        events contribute all of them). Also the size of the pooled comparison sample of each sample.
    Niter : int, default 150
        Number of SGD steps (no convergence check: inspect the plots shown with `verbose`).
    lr : float, default 0.5
        SGD learning rate (on the log factors).
    momentum : float, default 0.8
        SGD momentum.
    device : str, default "cuda:0"
        Torch device for the fit, e.g. ``"cpu"``, ``"cuda:0"``. On the CPU the fit is much slower (minutes per fit).
    verbose : bool, default True
        Show a tqdm progress bar and, after the fit, plot the loss and the log factors per step (``plt.show()``).
    init : {'null', 'mean'}, default 'null'
        Initial factors: 'null' starts all factors at 1, 'mean' at `fit_factor_MeanMatching` on all samples of each
        batch pooled.
    anchor : str, optional
        Batch whose factor is set to 1 after the fit (all factors are divided by its factor). Default: no anchor, the
        factors are normalised to a geometric mean of 1 over all batches.
    dummy : bool, default False
        Skip the fit (and all checks) and return 1 for every batch.
    name : str, default ""
        Label of the progress bar.

    Returns
    -------
    pandas.Series
        Factor per batch (index: the outer keys of `datadct`, in dict order). Multiply the linear data of every sample
        of a batch by its factor to standardise it.

    Raises
    ------
    ValueError
        If `anchor` is not a key of `datadct`, a (batch, sample) entry has no events, or no sample is present in more
        than one batch.
    NotImplementedError
        If `init` is not 'null' or 'mean'.

    Warns
    -----
    UserWarning
        If some batches share no sample with any other batch: their factors are not constrained by the data.

    See Also
    --------
    fit_factor_Sinkhorn : one sample per batch, also pairwise comparisons.

    Notes
    -----
    Samples present in only one batch carry no information and are ignored. The loss is the mean over all compared
    (batch, sample) entries of the Sinkhorn divergence to the pooled sample; loss settings, the mean-0 constraint on
    the log factors and the role of `anchor` are as in `fit_factor_Sinkhorn` (p=2, blur 0.5, reach 1, scaling 0.5,
    debiased; tuned for a 23-colour panel at cofactor 5000).

    Examples
    --------
    >>> factors = fit_factor_Sinkhorn_grouped({"day1": {"donorA": xa1, "donorB": xb1},
    ...                                        "day2": {"donorA": xa2, "donorB": xb2}}, anchor="day1")
    """
    # prepare data
    keys = np.asarray(list(datadct.keys()))
    Nbatch = len(keys)

    if dummy:
        return pd.Series(1., index=keys)
    
    if anchor is not None and not anchor in keys:
        raise ValueError(f"Anchor needs to be one of the data dict keys, '{anchor}' is not available! {list(keys)}")
    
    # flatten into (batch index, sample, data) entries, group entries by sample
    entries = [(i, sample, v) for i, k in enumerate(keys) for sample, v in datadct[k].items()]
    _check_nonempty({(k, sample):len(v) for k in keys for sample, v in datadct[k].items()})
    entry_batch = [e[0] for e in entries]
    samples = list(dict.fromkeys([e[1] for e in entries]))
    sample_entries = [[j for j, e in enumerate(entries) if e[1]==sample] for sample in samples]
    sample_entries = [se for se in sample_entries if len(se)>1]
    if len(sample_entries)==0:
        raise ValueError("No sample is present in more than one batch, the batch factors cannot be fitted!")
    batches_unshared = [keys[i] for i in range(Nbatch) if not any(entry_batch[j]==i for se in sample_entries for j in se)]
    if len(batches_unshared)>0:
        warnings.warn(f"Batches {batches_unshared} share no sample with other batches, their factors are not constrained by the fit!")
    
    data_torch_split = [torch.tensor(np.asarray(e[2]), dtype=torch.float32) for e in entries]
    data_torch = torch.cat(data_torch_split)[:,None].to(device)
    
    borders = np.cumsum([0]+[d.shape[0] for d in data_torch_split])
    samplerinds = [torch.arange(borders[i],borders[i+1]) for i in range(len(borders)-1)]
    sampler = Sampler(samplerinds, Nsamp, device=device)
    
    # prepare fit
    fct_loss = GeomSampleLoss("sinkhorn", blur=.5, reach=1.)
    fct_ash = lambda x: torch.arcsinh(x / arcsinh_cofactor)
    if init=="null":
        logfactor_init = torch.zeros((Nbatch,), dtype=torch.float32)
    elif init=="mean":
        datadct_pooled = {k:np.concatenate([np.asarray(v) for v in datadct[k].values()]) for k in keys}
        logfactor_init = torch.tensor(fit_factor_MeanMatching(datadct_pooled).loc[keys].to_numpy(), dtype=torch.float32).log()
    else:
        raise NotImplementedError(f"Initialisation '{init}' not available!")

    param_logfactor = ExtendableParameter((Nbatch,),0,0, allow_grad=True, axis_norm=0)
    param_logfactor.set_trainable_part(logfactor_init)
    param_logfactor.to(device)
    
    def _get_loss():
        inds, group = sampler.sample(split=True)
        factors = param_logfactor().exp()
        datas_scaled = [factors[entry_batch[j]] * data_torch[inds[j]] for j in range(len(entries))]

        # compare each entry to the aggregate of its sample across batches
        loss = 0.
        for se in sample_entries:
            data_comp = torch.cat([datas_scaled[j] for j in se], 0)
            data_comp = data_comp[torch.randperm(data_comp.shape[0])][:Nsamp]
            for j in se:
                loss += fct_loss(fct_ash(datas_scaled[j]), fct_ash(data_comp))
        loss /= sum([len(se) for se in sample_entries])
    
        return loss
    
    # fit
    optimizer = torch.optim.SGD(param_logfactor.parameters(), lr=lr, momentum=momentum)
    losshistory = []
    paramhistory = []

    with tqdm(range(Niter), desc=name, disable=not verbose) as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = _get_loss()
            loss.backward()
            losshistory.append(loss.item())
            paramhistory.append(param_logfactor().detach().cpu().numpy())
            optimizer.step()
    
    if verbose:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(losshistory)
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("OT Loss")
        ax[1].plot(np.asarray(paramhistory))
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("Scale Factor [log]")
        plt.show()

    factors = pd.Series(param_logfactor().exp().detach().cpu().numpy(), index=keys)
    # normalize here, fit more stable if only mean is fixed
    if anchor is not None:
        factors = factors / factors.loc[anchor]
        
    return factors

def fit_factors_Sinkhorn(datadct, fit_markers=None, arcsinh_cofactor=5000, Nsamp=2000, Niter=300, lr=5e-1, momentum=.8, device="cuda:0", verbose=True, aggregate=True, Ncomp=10, init="null", anchor=None, dummy=False, separate=False):
    """Fit one scaling factor per batch and marker by matching distributions with a Sinkhorn loss.

    The multi-marker version of `fit_factor_Sinkhorn`: all factors are fitted together, either with one Sinkhorn loss
    on the joint distribution of the fitted markers (default) or with one loss per marker (`separate`).

    Parameters
    ----------
    datadct : dict of str to pandas.DataFrame
        Events per batch, DataFrame of shape (events, stain) with intensities on a linear scale (raw or calibrated,
        unmixed; not arcsinh-transformed), e.g. ``{ad.uns["date"]: ad.obsm["unmx"] for ad in adatas}``. All frames
        must have the same, unique marker columns; they are aligned to the column order of the first frame (R2d-D1).
        The dict order sets the row order of the result.
    fit_markers : list of str, optional
        Markers that enter the loss. The others keep factor 1 (they are still returned). Names not among the columns
        are ignored silently; if no name matches, the fit fails with a torch RuntimeError. Default: all markers.
    arcsinh_cofactor : float, default 5000
        Cofactor (raw intensity units) of the transform used inside the loss: the loss compares
        ``arcsinh(x * factor / arcsinh_cofactor)`` between batches. The factors are applied to the linear data.
    Nsamp : int, default 2000
        Events drawn per batch in every step (a new random subsample each step; batches with fewer events contribute
        all of them). Also the size of the pooled comparison sample when `aggregate`.
    Niter : int, default 300
        Number of SGD steps (no convergence check: inspect the plots shown with `verbose`).
    lr : float, default 0.5
        SGD learning rate (on the log factors).
    momentum : float, default 0.8
        SGD momentum.
    device : str, default "cuda:0"
        Torch device for the fit, e.g. ``"cpu"``, ``"cuda:0"``. On the CPU the fit is much slower (minutes per fit).
    verbose : bool, default True
        Show a tqdm progress bar and, after the fit, plot the loss and the log factors per step (``plt.show()``).
    aggregate : bool, default True
        If True, compare every batch to a pooled sample (`Nsamp` events drawn from the rescaled subsamples of all
        batches, including the batch itself); the loss is the mean over batches. If False, compare pairs of batches
        directly (see `Ncomp`).
    Ncomp : int, default 10
        Only if not `aggregate`: number of random batch pairs compared per step (all pairs if there are fewer); the
        loss is the mean over these pairs.
    init : {'null', 'mean'}, default 'null'
        Initial factors: 'null' starts all factors at 1, 'mean' at `fit_factor_MeanMatching` per fitted marker (1 for
        the other markers).
    anchor : str, optional
        Batch whose factors are set to 1 after the fit (every marker column is divided by the anchor's factor).
        Default: no anchor, the factors of each marker are normalised to a geometric mean of 1 over all batches.
    dummy : bool, default False
        Skip the fit and return 1 everywhere (the marker columns are still checked and aligned).
    separate : bool, default False
        If False, one Sinkhorn loss on the joint (multi-dimensional) distribution of the fitted markers, so
        co-expression patterns enter the fit. If True, one 1-D Sinkhorn loss per fitted marker, summed over markers
        (the loss and so the effective step size grow with the number of fitted markers).

    Returns
    -------
    pandas.DataFrame
        Factors of shape (batch, stain): index the keys of `datadct`, columns the markers in the order of the first
        frame. Multiply each batch's linear data column-wise by its row to standardise it.

    Raises
    ------
    ValueError
        If the first frame has duplicate columns, a frame's marker set differs from the first frame's, `anchor` is not
        a key of `datadct`, or a batch has no events.
    NotImplementedError
        If `init` is not 'null' or 'mean'.

    See Also
    --------
    fit_factor_Sinkhorn : one marker; settings and loss details.
    ScalingTransformer : the same per-batch, per-marker scaling within the reference model (`OverlapFitModel`).

    Notes
    -----
    The log factors are constrained to mean 0 over batches for every marker; `anchor` only rescales the result.
    Loss settings are as in `fit_factor_Sinkhorn` (p=2, blur 0.5, reach 1, scaling 0.5, debiased; tuned for a
    23-colour panel at cofactor 5000). The aggregate loss is divided by the number of batches, as in
    `fit_factor_Sinkhorn`. Unfitted markers are scaled with their factor 1 but are not part of the loss.

    Examples
    --------
    >>> datadct = {ad.uns["date"]: ad.obsm["unmx"] for ad in adatas}
    >>> factors = fit_factors_Sinkhorn(datadct, fit_markers=["CD3", "CD19"], lr=2e-1, momentum=.7,
    ...                                aggregate=False, anchor="RF_12_11_2021")
    >>> standardised = {b: df[factors.columns] * factors.loc[b] for b, df in datadct.items()}
    """
    # prepare data
    keys = np.asarray(list(datadct.keys()))
    markers = next(iter(datadct.items()))[1].columns.to_list()
    Nbatch = len(keys)
    Nmarker = len(markers)
    marker_train = [(marker in fit_markers if fit_markers is not None else True) for marker in markers]
    # all frames must have the same markers; align them to the first frame's column order (tensors are positional)
    if len(set(markers)) != len(markers):
        raise ValueError(f"Duplicate marker columns in the data: {markers}")
    for k, v in datadct.items():
        if set(v.columns) != set(markers) or len(v.columns) != len(markers):
            raise ValueError(f"Batch '{k}' has markers {list(v.columns)}, expected the same set as the first batch: {markers}")
    datadct = {k: v[markers] for k, v in datadct.items()}
    
    if dummy:
        return pd.DataFrame(1., index=keys, columns=markers)
    
    if anchor is not None and not anchor in keys:
        raise ValueError(f"Anchor needs to be one of the data dict keys, '{anchor}' is not available! {list(keys)}")
    _check_nonempty({k:len(v) for k, v in datadct.items()})
    
    data_torch_split = [torch.tensor(v.to_numpy(), dtype=torch.float32) for v in datadct.values()]
    data_torch = torch.cat(data_torch_split).to(device)
    train_mask = torch.tensor(marker_train).to(device) #slice(None) #
    
    borders = np.cumsum([0]+[d.shape[0] for d in data_torch_split])
    samplerinds = [torch.arange(borders[i],borders[i+1]) for i in range(len(borders)-1)]
    sampler = Sampler(samplerinds, Nsamp, device=device)
    
    # prepare fit
    losscls = GeomSampleLoss("sinkhorn", blur=.5, reach=1.)
    if separate:
        fct_loss = lambda x, y: _sum_loss_separate_markers(losscls, x, y, None, None)
    else:
        fct_loss = lambda *args: losscls(*args)
    fct_ash = lambda x: torch.arcsinh(x / arcsinh_cofactor)
    if init=="null":
        logfactor_init = torch.zeros((Nbatch, Nmarker), dtype=torch.float32)
    elif init=="mean":
        def get_logfactor_mean(marker, dummy=False):
            return np.log(fit_factor_MeanMatching({k:v[marker] for k,v in datadct.items()}, dummy=dummy).loc[keys].to_numpy())
        logfactor_init = torch.tensor(np.vstack([get_logfactor_mean(m, dummy=not t) for m,t in zip(markers, marker_train)]).T, dtype=torch.float32)
    else:
        raise NotImplementedError(f"Initialisation '{init}' not available!")
    
    param_logfactor = ExtendableParameter((Nbatch, Nmarker),0,0, allow_grad=True, axis_norm=0)
    param_logfactor.set_trainable_part(logfactor_init)
    param_logfactor.to(device)
    
    def _get_loss():
        inds, group = sampler.sample(split=True)
        triu_comp = _triu_indices(Nbatch, Ncomp)
        factors = param_logfactor().exp()
        datas_scaled = [factors[i] * data_torch[inds[i]] for i in range(Nbatch)]
        
        if aggregate:
            loss = 0.
            data_comp = torch.cat(datas_scaled, 0)
            data_comp = data_comp[torch.randperm(data_comp.shape[0])][:Nsamp]
            for i in range(Nbatch):
                loss += fct_loss(fct_ash(datas_scaled[i][:,train_mask]), fct_ash(data_comp[:,train_mask]))
            loss /= len(datas_scaled)
        else:
            loss = 0.
            for i, j in zip(triu_comp[0].numpy(), triu_comp[1].numpy()):
                loss += fct_loss(fct_ash(datas_scaled[i][:,train_mask]), fct_ash(datas_scaled[j][:,train_mask]))
            loss /= triu_comp.shape[1]
    
        return loss
    
    optimizer = torch.optim.SGD(param_logfactor.parameters(), lr=lr, momentum=momentum)
    losshistory = []
    paramhistory = []
    
    with tqdm(range(Niter), disable=not verbose) as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = _get_loss()
            loss.backward()
            losshistory.append(loss.item())
            paramhistory.append(param_logfactor()[:,train_mask].detach().cpu().numpy())
            optimizer.step()
    
    if verbose:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(losshistory)
        ax[0].set_xlabel("Step")
        ax[0].set_ylabel("OT Loss")
        ax[1].plot(np.asarray(paramhistory).reshape(len(paramhistory), -1))
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("Scale Factor [log]")
        plt.show()
        
    factors = pd.DataFrame(param_logfactor().exp().detach().cpu().numpy(), index=keys, columns=markers)
    # normalize here, fit more stable if only mean is fixed
    if anchor is not None:
        factors = factors / factors.loc[anchor].to_numpy()[None]
  
    return factors

def fit_factor_MeanMatching(datadct, anchor=None, dummy=False):
    """Fit one scaling factor per batch by matching the batch means.

    Each factor scales its batch's mean (of the events clipped at 0) to the average of the batch means:
    ``factor_b = mean_k(m_k) / m_b``. Fast and deterministic, but sensitive to shifts in population composition.

    Parameters
    ----------
    datadct : dict of str to numpy.ndarray
        Events of one marker per batch: 1-D array (or pandas.Series) of intensities on a linear scale (raw or
        calibrated; not arcsinh-transformed).
    anchor : str, optional
        Batch whose factor is set to 1 (all factors are divided by its factor). Default: no anchor, the factors are
        such that every batch mean becomes the arithmetic mean of the batch means (not a geometric mean of 1).
    dummy : bool, default False
        Skip the computation and return 1 for every batch.

    Returns
    -------
    pandas.Series
        Factor per batch (index: the keys of `datadct`). Multiply the linear data of each batch by its factor to
        standardise it.

    Raises
    ------
    KeyError
        If `anchor` is not a key of `datadct` (raised by pandas).

    Warns
    -----
    UserWarning
        If a factor is not positive and finite (e.g. a batch whose clipped mean is 0).

    See Also
    --------
    fit_factor_PercentileMatching : matches a percentile instead of the mean.
    fit_factor_Sinkhorn : distribution matching; uses this function for ``init="mean"``.

    Notes
    -----
    Negative values (e.g. from unmixing) are set to 0 before the mean is taken.

    Examples
    --------
    >>> factors = fit_factor_MeanMatching({"a": xa, "b": xb}, anchor="a")
    """
    if dummy:
        return pd.Series(1., index=list(datadct.keys()))
    means = pd.Series({k:np.clip(v, a_min=0, a_max=None).mean() for k,v in datadct.items()})
    factors = means.mean() / means
    if np.any(factors<=0) or np.any(~np.isfinite(factors)):
        warnings.warn("Some fitted factors are negative, zero, or infinite, this is probably not desired!")
    if anchor is not None:
        factors = factors / factors.loc[anchor]
    return factors

def fit_factor_PercentileMatching(datadct, percentile=95, anchor=None, dummy=False):
    """Fit one scaling factor per batch by matching a percentile of the batch distributions.

    Each factor scales its batch's `percentile` to the average over batches: ``factor_b = mean_k(q_k) / q_b``.
    A high percentile matches the positive population of a marker, independent of the negative events.

    Parameters
    ----------
    datadct : dict of str to numpy.ndarray
        Events per batch, array-like of intensities on a linear scale (raw or calibrated; not arcsinh-transformed).
        Arrays with more than one dimension are flattened, i.e. all their values are pooled.
    percentile : float, default 95
        Percentile (0-100) to match. No clipping: it should lie in the positive range of all batches.
    anchor : str, optional
        Batch whose factor is set to 1 (all factors are divided by its factor). Default: no anchor, the factors are
        such that every batch percentile becomes the arithmetic mean of the batch percentiles.
    dummy : bool, default False
        Skip the computation and return 1 for every batch.

    Returns
    -------
    pandas.Series
        Factor per batch (index: the keys of `datadct`). Multiply the linear data of each batch by its factor to
        standardise it.

    Raises
    ------
    KeyError
        If `anchor` is not a key of `datadct` (raised by pandas).

    Warns
    -----
    UserWarning
        If a factor is not positive and finite (e.g. a percentile <= 0); the message includes the factors.

    See Also
    --------
    fit_factor_MeanMatching : matches the mean instead.

    Examples
    --------
    >>> factors = fit_factor_PercentileMatching({b: df["CD3"].to_numpy() for b, df in unmx_by_batch.items()}, 97)
    """
    if dummy:
        return pd.Series(1., index=list(datadct.keys()))
    percentiles = pd.Series({k:np.percentile(np.asarray(v).ravel(), percentile) for k,v in datadct.items()})
    factors = percentiles.mean() / percentiles
    if np.any(factors<=0) or np.any(~np.isfinite(factors)):
        warnings.warn("Some fitted factors are negative, zero, or infinite, this is probably not desired!\n"+str(factors))
    if anchor is not None:
        factors = factors / factors.loc[anchor]
    return factors
