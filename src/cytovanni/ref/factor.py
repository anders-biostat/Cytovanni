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
    ind_triu = torch.triu_indices(N,N, offset=1)
    ind_triu = ind_triu[:,~torch.all(torch.isin(ind_triu, torch.tensor(exclude_internal)), axis=0)]
    if Nmax is None:
        return ind_triu
    else:
        return ind_triu[:,torch.randperm(ind_triu.shape[1])][:,:Nmax]

def fit_factor_Sinkhorn_LBFGS(datadct, arcsinh_cofactor=5000, Nsamp=2000, Niter=10, device="cuda:0", verbose=False, aggregate=True, Ncomp=10, init="null", anchor=None, name=""):
    """ Fit scaling factor using Sinkhorn divergence as the loss.

        Optimized using LBFGS, a fancy version of Newton's method.
    """
    # prepare data
    keys = np.asarray(list(datadct.keys()))
    Nbatch = len(keys)
    
    if anchor is not None and not anchor in keys:
        raise ValueError(f"Anchor needs to be one of the data dict keys, '{anchor}' is not available! {list(keys)}")
    
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
        raise NotImplemented(f"Initialisation '{init}' not available!")

    param_logfactor = ExtendableParameter((Nbatch,),0,0, allow_grad=True, axis_norm=0)
    param_logfactor.set_trainable_part(logfactor_init)
    param_logfactor.to(device)

    inds, group = sampler.sample(split=True)
    
    def _get_loss():
        triu_comp = _triu_indices(Nbatch, Ncomp)
        factors = param_logfactor().exp()
        datas_scaled = [factors[i] * data_torch[inds[i]] for i in range(Nbatch)]

        if aggregate:
            loss = 0.
            data_comp = torch.cat(datas_scaled, 0)
            data_comp = data_comp[torch.randperm(data_comp.shape[0])][:Nsamp]
            for i in range(Nbatch):
                loss += fct_loss(fct_ash(datas_scaled[i]), fct_ash(data_comp))
            loss /= triu_comp.shape[1]
        else:
            loss = 0.
            for i, j in zip(triu_comp[0].numpy(), triu_comp[1].numpy()):
                loss += fct_loss(fct_ash(datas_scaled[i]), fct_ash(datas_scaled[j]))
            loss /= triu_comp.shape[1]
    
        return loss
    
    # fit
    optimizer = torch.optim.LBFGS(param_logfactor.parameters(), lr=1.)
    losshistory = []
    paramhistory = []

    def closure():
        optimizer.zero_grad()
        loss = _get_loss()
        loss.backward()
        return loss

    def log_loss():
        with torch.no_grad():
            loss = _get_loss()
            losshistory.append(loss.item())
            paramhistory.append(param_logfactor().detach().cpu().numpy())

    with tqdm(range(Niter), desc=name, disable=not verbose) as pbar:
        for i in pbar:
            log_loss()
            optimizer.step(closure)
            inds, group = sampler.sample(split=True)
    
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

def fit_factor_Sinkhorn(datadct, arcsinh_cofactor=5000, Nsamp=2000, Niter=150, lr=5e-1, momentum=.8, device="cuda:0", verbose=True, aggregate=True, Ncomp=10, init="null", anchor=None, dummy=False, name=""):
    """ Fit scaling factor using Sinkhorn divergence as the loss.

        Uses normal SGD.
    """
    # prepare data
    keys = np.asarray(list(datadct.keys()))
    Nbatch = len(keys)

    if dummy:
        return pd.Series(1., index=keys)
    
    if anchor is not None and not anchor in keys:
        raise ValueError(f"Anchor needs to be one of the data dict keys, '{anchor}' is not available! {list(keys)}")
    
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
        raise NotImplemented(f"Initialisation '{init}' not available!")

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
            loss /= triu_comp.shape[1]
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
    """ Fit scaling factor using Sinkhorn divergence as the loss, with multiple samples per batch.

        Like fit_factor_Sinkhorn, but datadct is {batch: {sample: data}}, with one shared factor per batch.
        Each sample is only compared to the aggregate of the same sample across batches, so differences
        in composition or cell number between samples do not enter the fit.
        Samples present in only one batch carry no information and are ignored.

        Uses normal SGD.
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
    entry_batch = [e[0] for e in entries]
    samples = list(dict.fromkeys([e[1] for e in entries]))
    sample_entries = [[j for j, e in enumerate(entries) if e[1]==sample] for sample in samples]
    sample_entries = [se for se in sample_entries if len(se)>1]
    
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
        raise NotImplemented(f"Initialisation '{init}' not available!")

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
    """ Fit scaling factors using Sinkhorn divergence as the loss.

        Basically fit_factor_Sinkhorn, but for multiple factors in parallel.
        Requires all datas to have all markers.

        Uses normal SGD.

        :param datadct: dict. Dictionary that contains the data to be fitted, should have the batch name as key, and pd.DataFrame with index cell and column marker as value for each batch.

        :param arcsinh_cofactor: float. Arcsinh cofactor to use during the fit.

        :param Nsamp: int. Number of cells to sample per batch for the Sinkhorn loss.

        :param Niter: int. Number of fit iterations.

        :param lr: float. Learning rate.

        :param momentum: float. Momentum.

        :param device: str. Device (cpu, cuda:0 etc.) to perform the fit on.

        :param verbose: bool. If true, also plots the loss etc.

        :param aggregate: bool. If true, calculates the loss between each batch and their aggregate, if false performs pairwise comparisons.

        :param Ncomp: int. If not aggregate, number of total comparisons per iteration.

        :param init: str. Either 'null', i.e. initializing all factors as 1., or 'mean', which uses mean matching as the initialization.

        :param anchor: None, str. If given, normalizes factors such that this batch stays the same.

        :param dummy: bool. If True, skips the fit and returns 1. for all scaling factors.

        :param separate: bool. If True, uses Sinkhorn separately on each parameter, instead of on all of them at once.
    """
    # prepare data
    keys = np.asarray(list(datadct.keys()))
    markers = next(iter(datadct.items()))[1].columns.to_list()
    Nbatch = len(keys)
    Nmarker = len(markers)
    marker_train = [(marker in fit_markers if fit_markers is not None else True) for marker in markers]
    
    if dummy:
        pd.DataFrame(1., index=keys, columns=markers)
    
    if anchor is not None and not anchor in keys:
        raise ValueError(f"Anchor needs to be one of the data dict keys, '{anchor}' is not available! {list(keys)}")
    
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
        raise NotImplemented(f"Initialisation '{init}' not available!")
    
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
    """ Fit factor using mean matching.
        
        Clips negative values to zero before calculating the mean, not sure this is the best choice.
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
    """ Fit factor using percentile matching.
    """
    if dummy:
        return pd.Series(1., index=list(datadct.keys()))
    percentiles = pd.Series({k:np.percentile(v.flatten(), percentile) for k,v in datadct.items()})
    factors = percentiles.mean() / percentiles
    if np.any(factors<=0) or np.any(~np.isfinite(factors)):
        warnings.warn("Some fitted factors are negative, zero, or infinite, this is probably not desired!\n"+str(factors))
    if anchor is not None:
        factors = factors / factors.loc[anchor]
    return factors
