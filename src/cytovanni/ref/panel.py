import torch
from torch.nn import Module, Parameter, ParameterDict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..spill import PanelConfiguration
from ..exceptions import SpilloverInversionException, OverlapStandardisationException, OverlapStandardisationWarning
from ..torch import ExtendableParameter, diffclamp
from ..torch import torch_invert_spectra, torch_apply_unmixing_inv, torch_apply_unmixing_lstsq
from ..torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict
from .moe import MultiOrdinalEncoder


class PanelModule(Module):
    """ Module to hold the spectra of the panel,
        dealing with all the complications of turning an embedding vector into a spectrum matrix.
        All parameters are set initially and not trained.
        
        Automatically drops all unused components, but still expects them to be present in the embedding.
        I.e. if there are 130 total PCs, and only 40 are used, the embedding should still have 130 entries.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_panelconfig(cls, panelconfig):
        """ Initialise using PanelConfiguration data.
        """
        pm = cls()
        pm.panelconfig = panelconfig
        
        pm.channels = pm.panelconfig.pca_components.columns.to_numpy()
        pm.components = pm.panelconfig.pca_components.index.to_numpy()
        pm.stains = pm.panelconfig.pca_offset.index.to_numpy()
        
        # general offset (stain, channel)
        pm.param_offset = Parameter(torch.tensor(pm.panelconfig.pca_offset.to_numpy(), dtype=torch.float32), requires_grad=False)
        
        # principal components (stain, channel, component)
        param_components = np.zeros((len(pm.channels), len(pm.channels), len(pm.components)))
        stainind = pm.panelconfig.pca_offset.index.get_indexer(pm.panelconfig.pca_components_stain)
        param_components[stainind, :, np.arange(stainind.shape[0])] = pm.panelconfig.pca_components.to_numpy()
        pm.param_components = Parameter(torch.tensor(param_components, dtype=torch.float32), requires_grad=False)
        
        # component mask (1, 1, component)
        # don't include this in above to later do other stuff with the unused components
        mask = pm.panelconfig.pca_components_use.to_numpy().astype(float)
        pm.param_components_mask = Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
        
        return pm
    
    @classmethod
    def from_exported(cls, exported):
        """ Initialise from exported data.
        """
        panelconfig = PanelConfiguration.from_saved(exported=exported["panelconfig"])
        pm = cls.from_panelconfig(panelconfig)
        pm.load_state_dict(exported["state_dict"])
        
        return pm
    
    def export(self):
        exported = {}
        exported["panelconfig"] = self.panelconfig.export()
        exported["state_dict"] = self.state_dict()
        return exported
    
    def requires_grad_(self, requires_grad=True):
        """ Set requires_grad properly on all parameters.
        """
        self.param_offset.requires_grad_(False)
        self.param_components.requires_grad_(False)
        self.param_components_mask.requires_grad_(False)
    
    def forward(self, x_embedding, clamp_differentiable=True, mask_unused=True):
        """ Transform embedding into spectra matrix.
            x_embedding should have shape (components, ...)
            returns shape (..., stain, channel)
            
            Clamps spectra matrix to positive, but still passes gradients through clamped parts if clamp_differentiable is True.
            Setting mask_unused to False disables masking of unused components.
        """
        offset = self.param_offset.view( [1]*(len(x_embedding.shape)-1) + list(self.param_offset.shape) )
        comp = torch.einsum("ijk, k... -> ...ij", self.param_components * (self.param_components_mask if mask_unused else 1.), x_embedding)
        if clamp_differentiable:
            return diffclamp(offset + comp, 0, None)
        else:
            return (offset + comp).clamp(0, None)
    
    def apply_component_mask(self, emb):
        """ Apply component mask to emb of shape (components, ...).
        """
        mask = self.param_components_mask.view([len(self.param_components_mask)]+list(emb.shape[1:]))
        return mask * emb

    def get_jacobian_ashunmx_jacrev(self, x, embedding, include_unused=True, arcsinh_cofactor=1500):
        """ Takes raw intensities (..., channel) and embedding (components), as torch tensors,
            returns Jacobian of ArcSinh transformed values (..., stain, components).

            Output is on same device as self.

            Uses jacrev, simpler to implement but very large memory footprint.
            Only use to check efficient implementation!

            :param x: torch.Tensor. Raw intensities, shape (..., channel).

            :param embedding: torch.Tensor. Component embedding, shape (components).

            :param include_unused: bool. Whether to include unused components of the panel.

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.
        """
        x, embedding = x.to(self.device), embedding.to(self.device)
        with torch.no_grad():
            fct_transform = lambda x, emb: (torch_apply_unmixing_inv(
                                                torch_invert_spectra(
                                                    self(emb, clamp_differentiable=False, mask_unused=not include_unused)
                                                        ), x) / arcsinh_cofactor).arcsinh()
            jacob = torch.func.jacrev(lambda emb: fct_transform(x, emb), chunk_size=None)(embedding)
        torch.cuda.empty_cache()
        return jacob

    def get_jacobian_ashunmx(self, x, embedding, stainscale=None, include_unused=True, arcsinh_cofactor=1500):
        """ Takes raw intensities (..., channel) and embedding (components), as torch tensors,
            returns Jacobian of ArcSinh transformed values (..., stain, components).

            Output is on same device as self.

            Much more efficient implementation than the version that uses jacrev.

            :param x: torch.Tensor. Raw intensities, shape (..., channel).

            :param embedding: torch.Tensor. Component embedding, shape (components).
            
            :param stainscale: None, torch.Tensor. Optional stain scaling, shape (..., stain)

            :param include_unused: bool. Whether to include unused components of the panel.

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.
        """
        x, embedding, stainscale = x.to(self.device), embedding.to(self.device), stainscale.to(self.device) if stainscale is not None else 1.
        with torch.no_grad():
            fct_inv_spectra = lambda emb: torch_invert_spectra(self(emb, clamp_differentiable=False, mask_unused=not include_unused))
            inv_spectra = fct_inv_spectra(embedding) # (channel, stain)
            inv_spectra_jac = torch.func.jacrev(fct_inv_spectra, chunk_size=None)(embedding) # (channel, stain, components)

            x_unmx = torch_apply_unmixing_inv(inv_spectra, x) # (..., stain)
            d_arcsinh = torch.pow((x_unmx * stainscale / arcsinh_cofactor).square() +1, -.5) * stainscale / arcsinh_cofactor # (..., stain)

            d_unmx = torch.einsum("ijc, ...i -> ...jc", inv_spectra_jac, x) # (..., stain, components)

            jacob = (d_arcsinh[...,None] * d_unmx) # (..., stain, components)
        return jacob
    
    def apply_ashunmx(self, x, embedding, stainscale=None, include_unused=True, arcsinh_cofactor=1500):
        """ Takes raw intensities (..., channel) and embedding (..., components), as torch tensors,
            returns unmixed and ArcSinh transformed values (..., stain).
            
            x and embedding need to be broadcastable.

            Output is on same device as self.

            :param x: torch.Tensor. Raw intensities, shape (..., channel).

            :param embedding: torch.Tensor. Component embedding, shape (..., components).
            
            :param stainscale: None, torch.Tensor. Optional stain scaling, shape (..., stain)

            :param include_unused: bool. Whether to include unused components of the panel.

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.
        """
        x, embedding, stainscale = x.to(self.device), embedding.to(self.device), stainscale.to(self.device) if stainscale is not None else 1.
        inv_spectra = torch_invert_spectra(self(embedding, clamp_differentiable=False, mask_unused=not include_unused))
        return (torch_apply_unmixing_inv(inv_spectra, x) * stainscale / arcsinh_cofactor).arcsinh()
    
    def get_unmixing_resistance(self, x_np, embedding_np, stainscale_np=None, arcsinh_cofactor=1500, embedding_step=1., include_unused=False, single_step=True, gradstep_maxnorm=True, batch_size=1e4):
        """ Get resistance to unmixing errors R.

            :param x_np: np.array. Raw intensities, shape (events, channel).

            :param embedding_np: np.array. Panel embedding, shape (components).
            
            :param stainscale_np: None, np.array. Optional stain scaling, shape (stain).

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.

            :param embedding_step: float. How much to +- vary the embedding.

            :param include_unused: bool. Whether to include PCA components that are not used for the panel in the calculation.

            :param single_step: bool. If True, uses single step along most important component, otherwise along gradient with most important component changed by embedding_step.
            
            :param gradstep_maxnorm: bool. If True, normalize max change to embedding_step, else mean change.

            :param batch_size: int, float. Perform in chunks of size batch_size to limit memory use, no point in going above 1e4.
        """
        x_np_batches = np.array_split(x_np, int(np.ceil(x_np.shape[0] / batch_size)))
        embedding = torch.as_tensor(embedding_np, dtype=torch.float32).to(self.device)
        stainscale = torch.as_tensor(stainscale_np, dtype=torch.float32)[None].to(self.device) if stainscale_np is not None else None
        val = []
        val_std = []

        def get_std_singlestep(x, embedding, jacob, val):
            """ single step along highest jacobian component
            """
            # max component for every stain
            ind_maxcomp = jacob.abs().argmax(-1)
            N = ind_maxcomp.shape[0]
            Nstain = ind_maxcomp.shape[1]
            arNstain = torch.arange(Nstain)[None].to(self.device).expand(N, Nstain)
            arNstain_single = torch.arange(Nstain).to(self.device)
            arN = torch.arange(N)[:,None].to(self.device).expand(N, Nstain)
            # get modified embedding
            embedding_mod = embedding[:,None,None,None].repeat(1,N,Nstain,2)
            embedding_mod[ind_maxcomp.flatten(), arN.flatten(), arNstain.flatten(),:] = (torch.ones((Nstain*N,2)) * torch.tensor([[embedding_step,-embedding_step]], dtype=torch.float32)).to(self.device)
            # modified intensities, shape (event, maxcomp, 2, stain)
            val_mod = self.apply_ashunmx(x[:,None,None,:], embedding_mod, stainscale=stainscale[:,None,None,:] if stainscale is not None else None,
                                           include_unused=include_unused, arcsinh_cofactor=arcsinh_cofactor)
            val_mod = val_mod[:,arNstain_single,:,arNstain_single].swapaxes(0,1)
            val_std = (val[...,None] - val_mod).abs().mean(-1)
            return val_std

        def get_std_gradstep(x, embedding, jacob, val):
            """ step along gradient, normed to one for most important component
            """
            if gradstep_maxnorm:
                direction = jacob / jacob.abs().max(-1, keepdims=True).values
            else:
                direction = jacob / jacob.abs().sum(-1, keepdims=True)
            embedding_shift = (direction.movedim(-1,0)[...,None] * torch.tensor([embedding_step,-embedding_step], dtype=torch.float32).to(self.device)[None,None,None])
            embedding_mod = embedding[:,None,None,None] + embedding_shift
            # modified intensities, shape (event, maxcomp, 2, stain)
            val_mod = self.apply_ashunmx(x[:,None,None,:], embedding_mod, stainscale=stainscale[:,None,None,:] if stainscale is not None else None,
                                           include_unused=include_unused, arcsinh_cofactor=arcsinh_cofactor)
            arNstain_single = torch.arange(val_mod.shape[-1]).to(self.device)
            val_mod = val_mod[:,arNstain_single,:,arNstain_single].swapaxes(0,1)
            val_std = (val[...,None] - val_mod).abs().mean(-1)
            return val_std

        with torch.no_grad():
            for batch in x_np_batches:
                x = torch.as_tensor(batch, dtype=torch.float32).to(self.device)
                jacob = self.get_jacobian_ashunmx(x, embedding, include_unused=include_unused, arcsinh_cofactor=arcsinh_cofactor)
                t_val = self.apply_ashunmx(x, embedding, stainscale=stainscale if stainscale is not None else None,
                                           include_unused=include_unused, arcsinh_cofactor=arcsinh_cofactor)
                t_val_std = get_std_singlestep(x, embedding, jacob, t_val) if single_step else get_std_gradstep(x, embedding, jacob, t_val)
                val.append(t_val.cpu().numpy())
                val_std.append(t_val_std.cpu().numpy())
                del x, jacob, t_val, t_val_std # make sure memory is freed
        del embedding, stainscale # make sure memory is freed
        torch.cuda.empty_cache()
        val, val_std = np.vstack(val), np.vstack(val_std)
        R = np.arcsinh(np.abs(val/val_std)/20.)

        return val, val_std, R

    def get_unmixing_resistance_fromad(self, adata, embedding_np, key_layer="calibrated", arcsinh_cofactor=1500, embedding_step=1., include_unused=False, single_step=False, gradstep_maxnorm=True, batch_size=1e4, label_marker=True):
        """ Get resistance to unmixing errors R directly from adata.

            :param adata: AnnData. adata from which to get the data.

            :param embedding_np: np.array. Panel embedding, shape (components).

            :param key_layer: str. Key for raw intensities in adata.layer.

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.

            :param embedding_step: float. How much to +- vary the embedding.

            :param include_unused: bool. Whether to include PCA components that are not used for the panel in the calculation.

            :param single_step: bool. If True, uses single step along most important component, otherwise along gradient with most important component changed by embedding_step.
            
            :param gradstep_maxnorm: bool. If True, normalize max change to embedding_step, else mean change.

            :param batch_size: int, float. Perform in chunks of size batch_size to limit memory use, no point in going above 1e4.

            :param label_marker: bool. If True, label output with panelconfig.stain_marker_name, else with stain names.
        """
        x_np = adata[:,self.channels].layers[key_layer]
        R = self.get_unmixing_resistance(x_np, embedding_np, arcsinh_cofactor=arcsinh_cofactor, embedding_step=embedding_step,
                                         include_unused=include_unused, single_step=single_step, gradstep_maxnorm=gradstep_maxnorm,
                                         batch_size=batch_size)[-1]
        return pd.DataFrame(R, index=adata.obs.index, columns=self.panelconfig.stain_marker_name if label_marker else self.stains)

    def get_grouped_jac_fromad(self, adata, embedding_np, stainscale_np=None, key_layer="calibrated", arcsinh_cofactor=1500, batch_size=1e4, label_marker=True):
        """ Get influence of all panel components on marker means for adata.
            Subset adata if only a subset of events should be used!

            Calculates the gradient of unmixed, ArcSinh transformed value for all events, returns mean across events.

            :param adata: AnnData. adata from which to get the data.

            :param embedding_np: np.array. Panel embedding, shape (components).
            
            :param stainscale_np: None, np.array. Optional stain scaling, shape (stain).

            :param key_layer: str. Key for raw intensities in adata.layer.

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.

            :param batch_size: int, float. Perform in chunks of size batch_size to limit memory use, no point in going above 1e4.

            :param label_marker: bool. If True, label output with panelconfig.stain_marker_name, else with stain names.
        """
        x_np = adata[:,self.channels].layers[key_layer]
        x_np_batches = np.array_split(x_np, int(np.ceil(x_np.shape[0] / batch_size)))
        embedding = torch.as_tensor(embedding_np, dtype=torch.float32).to(self.device)
        stainscale = torch.as_tensor(stainscale_np, dtype=torch.float32)[None].to(self.device) if stainscale_np is not None else None
        jacob = []
        with torch.no_grad():
            for batch in x_np_batches:
                x = torch.as_tensor(batch, dtype=torch.float32).to(self.device)
                jacob.append(self.get_jacobian_ashunmx(x, embedding, stainscale=stainscale,
                                                       include_unused=True, arcsinh_cofactor=arcsinh_cofactor).cpu().numpy())
        mean_jac = np.vstack(jacob).mean(0)
        return pd.DataFrame(mean_jac, index=self.panelconfig.stain_marker_name if label_marker else self.stains, columns=self.components).T
    
    def get_grouped_jac_single_fromad(self, adata, stain, key_layer="calibrated", arcsinh_cofactor=1500, s_dig=3, include=10):
        """ Run get_grouped_jac_fromad on adata after transformer.add_standardisation_parameters was called on it.
            Focus on one stain and also return which components were used in the panel.
            
            :param adata: AnnData. adata to get the data from.
            
            :param stain: str. Stain to focus on, in terms of panelconfig.stain_marker_name.
            
            :param key_layer: str. Raw data layer to use.
            
            :param arcsinh_cofactor: float. ArcSinh cofactor.
            
            :param s_dig: int. Number of significant digits to include in the Jacobian.
            
            :param include: int. Only keep the 'include' highest components.
        """
        jac = self.get_grouped_jac_fromad(adata,
                                    embedding_np=adata.uns["panel_embedding"], stainscale_np=adata.varm["sfm_stain_factor"].iloc[0].to_numpy(),
                                    key_layer=key_layer, arcsinh_cofactor=arcsinh_cofactor, batch_size=1e4, label_marker=True)
        jac_marker = np.round(jac[stain], s_dig)

        df = pd.DataFrame([jac_marker.abs(), jac_marker, self.panelconfig.pca_components_use], index=["abs. Jacobian", "Jacobian", "in Panel"]).T.sort_values("abs. Jacobian", ascending=False)
        df["in Panel"] = df["in Panel"].astype(bool)
    
        def highlight_unused(s):
            return ['background-color: orange' if not s["in Panel"] else 'background-color: lightgreen' for v in s]
        fct_style = lambda df: df.style.apply(highlight_unused, axis=1)

        if include is not None:
            df = df[:include]

        return df, fct_style
    
    
    def get_spectra_fromembedding(self, embedding, label_marker=True, include_unused=False):
        """ Get spectra DataFrame from embedding.

            :param embedding: iterable. Embedding, shape (components).

            :param label_marker: bool. If True, label output with panelconfig.stain_marker_name, else with stain names.

            :param include_unused: bool. Whether to include PCA components that are not used for the panel in the calculation.
        """
        with torch.no_grad():
            spectra = pd.DataFrame(self(torch.tensor(embedding, dtype=torch.float32).to(self.device)).detach().cpu().numpy(),
                                 index=self.panelconfig.stain_marker_name if label_marker else self.stains, columns=self.channels)
        return spectra
    
    def get_compmean_errorbars(self, x_np, embedding_np, arcsinh_cofactor=1500, embedding_step=1., label_marker=True, use_median=False, include_unused=False):
        """ Estimate compensation error bars for stain means.

            For every stain, takes the panel component that affects the (ArcSinh) mean value most and varies it by +-embedding_step.
            Returns a pd.DataFrame with the mean 'mean', extimated std 'std', and z score (mean/std) 'z'.
            
            Calculates the gradient in one pass through; for 110 PC components and 23 markers this needs around 1GB of VRAM per 100 events!

            :param x: array. Raw intensity, shape (event, channel).

            :param embedding: array. Panel embedding, shape (components).

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.

            :param embedding_step: float. How much to +- vary the embedding.

            :param label_marker: bool. If True, label output with panelconfig.stain_marker_name, else with stain names.
            
            :param use_median: bool. If True, use median instead of mean value. Still gets names 'mean' in the output for consistency.
            
            :param include_unused: bool. Whether to include PCA components that are not used for the panel in the calculation.
        """
        with torch.no_grad():
            # turn into tensors
            x = torch.as_tensor(x_np, dtype=torch.float32).to(self.device)
            embedding = torch.as_tensor(embedding_np, dtype=torch.float32).to(self.device)
            # function to get cluster means of unmixed
            fct_reduce = lambda x: x.mean(-2) if not use_median else x.median(-2).values
            fct_mean = lambda x, emb: fct_reduce((torch_apply_unmixing_inv(torch_invert_spectra(
                                                    self(emb, clamp_differentiable=False, mask_unused=not include_unused)
                                                    ), x) / arcsinh_cofactor).arcsinh())
            # get Jacobian of cluster means
            jacob = torch.func.jacrev(lambda emb: fct_mean(x, emb))(embedding)
            val = fct_mean(x, embedding)
            # max component for every stain
            ind_maxcomp = jacob.abs().argmax(-1)
            Nstain = len(ind_maxcomp)
            arN = torch.arange(Nstain).to(self.device)
            # get modified embedding
            embedding_mod = embedding[:,None,None].repeat(1,Nstain,2)
            embedding_mod[ind_maxcomp,arN,:] += (torch.ones((Nstain,2)) * torch.tensor([[embedding_step,-embedding_step]], dtype=torch.float32)).to(self.device)
            # modified means, shape (maxcomp, 2, stain)
            val_mod = fct_mean(x, embedding_mod[...,None])
            val_mod = val_mod[arN, :, arN]

            val_std = (val[:,None] - val_mod).abs().mean(-1)

            val, val_std = val.cpu().numpy(), val_std.cpu().numpy()

        df = pd.DataFrame([val, val_std, val/val_std], index=["mean", "std", "z"],
                          columns=self.panelconfig.stain_marker_name if label_marker else self.stains).T

        return df
    
    def get_clustermeans_withz(self, embedding, adata, key_cluster, key_layer="calibrated", cluster_reorder=True, Nmax=4000, arcsinh_cofactor=1500, embedding_step=1., label_marker=True, include_unused=True):
        """ Takes embedding and adata. Returns mean (on ArcSinh transformed markers),
            as well as z-transformed mean to indicate how different from zero the mean is if potential
            compensation errors are taken into account.

            For every stain and cluster, takes the panel component that affects the (ArcSinh) mean value most and
            varies it by +-embedding_step to get a sense of the potential standard deviation.

            :param embedding: array. Panel embedding, shape (components).

            :param key_cluster: str. Key of cluster in adata.obs.

            :param key_layer: str. Key of raw intensities for unmixing in adata.layers.

            :param cluster_reorder: bool. If True, use sns.clustermap output to reorder indices and rows in output.

            :param Nmax: int. Subsample to max number of events to use, no point in running this for a very large number of events.

            :param arcsinh_cofactor: float. Cofactor for ArcSinh transformation.

            :param embedding_step: float. How much to +- vary the embedding.

            :param label_marker: bool. If True, label output with panelconfig.stain_marker_name, else with stain names.

            :param include_unused: bool. Whether to include PCA components that are not used for the panel in the calculation.
        """
        clusters = np.unique(adata.obs[key_cluster])
        outs = []
        for c in clusters:
            X = adata[:,self.channels].layers[key_layer][adata.obs[key_cluster].to_numpy()==c]
            X = X[np.random.permutation(X.shape[0])[:Nmax]]
            outs.append(self.get_compmean_errorbars(X, embedding,
                                                    arcsinh_cofactor=arcsinh_cofactor, embedding_step=embedding_step,
                                                    label_marker=label_marker, include_unused=include_unused))
        means = pd.DataFrame([o["mean"] for o in outs], index=clusters)
        zs = pd.DataFrame([o["z"] for o in outs], index=clusters)

        if cluster_reorder:
            cm = sns.clustermap(means)
            plt.close()
            xind, yind = cm.dendrogram_row.reordered_ind, cm.dendrogram_col.reordered_ind
            means = means.iloc[xind].iloc[:,yind]
            zs = zs.iloc[xind].iloc[:,yind]

        return means, zs
    
    
    @property
    def device(self):
        return self.param_components_mask.device

class PanelEmbeddingModule(Module):
    """ Potentially initializes all embedding parameters as non-zero and trainable,
        however the PanelModule automatically only includes the used components.
    """
    TRAINED_EMBEDDING = True
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_data(cls, panelconfig, moe, key_comp_batch, batch_order=None, anchor_bidx={}, fixed_init=None, init_emb=None, max_abs_embedding=6):
        """ 
            :params panelconfig: PanelConfig. Config for the panel PCA embedding.
            
            :param moe: MultiOrdinalEncoder. Batch encoder for all relevant batch keys.
            
            :param key_comp_batch: pd.Series. Map which parameters should be fitted for which batch, index panel components, entry batch key. E.g. pd.Series(["a", "b"], index=["PC0","PC1"]) if batch 'a' should affect the embedding for 'PC0', and batch 'b' for 'PC1'.
            
            :param batch_order: None, iterable. If given, assumes that batches have a tree structure and trains common factors as well as separate ones. E.g. for the example above, if batch_order=["a", "b"], "b" will still have its own parameters for the embedding of 'PC1', but instead of "a" not affecting it it also has an additive factor for 'PC1'. Done such that if there are many different batches in "b" for every batch in "a", which should have more similar embeddings than batches that have a different "a", it speeds up the training if every different batch in "b" also trains a common factor for its batch in "a".
            
            :params anchor_bidx: dict. Anchor all batch indices in dict to not be trainable. E.g. {"a":[0,1]} disables training for all parameters associated with indices [0,1] in batch "a".
            
            :params fixed_init: None, str. If none, initializes all embeddings as zero. Otherwise initializes embeddings from compensation run fixed_init.
            
            :params init_emb: None, iterable. If given, initialize all embeddings with this.
            
            :params max_abs_embedding: None, float. Maximum absolute value for all embeddings, gets clipped above/below.
        """
        pem = cls()
        pem.panelconfig = panelconfig
        pem.panelmodule = PanelModule.from_panelconfig(pem.panelconfig)
        pem.init_add_from_panelmoduleconfig()
        pem.moe = moe
        pem.key_comp_batch = key_comp_batch
        pem.batch_order = batch_order
        pem.batch_keys = list(pem.key_comp_batch.unique())
        pem.max_abs_embedding = max_abs_embedding
        
        if init_emb is None:
            if fixed_init is None:
                pem.init_emb = torch.zeros((len(pem.panelmodule.components),1), dtype=torch.float32)
            else:
                pem.init_emb = torch.as_tensor(
                    pem.panelmodule.panelconfig.pca_embedding.loc[fixed_init].fillna(0).to_numpy(),
                    dtype=torch.float32)[:,None]
        else:
            pem.init_emb = torch.as_tensor(np.asarray(init_emb), dtype=torch.float32)[:,None]
        
        if not np.all(np.isin(pem.batch_keys, list(pem.moe.keys()))):
            missing = list(set(pem.batch_keys) - set(pem.moe.keys()))
            raise OverlapStandardisationException(f"Batches {missing} from the component batch mapping are not present in the batch encoder!")
        pem.key_comp_batch = pem.key_comp_batch.loc[pem.panelmodule.components]
        
        if batch_order is not None and len(set(pem.batch_keys)-set(pem.batch_order))>0:
            missing = set(pem.batch_keys)-set(pem.batch_order)
            raise OverlapStandardisationException(f"Batches {missing} from the component batch mapping are not present in the batch ordering! Either pass None, or provide an ordering of all used batches.")
        
        pem.init_params()
        pem.anchor_batches(anchor_bidx)
        
        return pem

    def init_add_from_panelmoduleconfig(self):
        """ Add some parameters from panelmodule and panelconfig to self.
        """
        self.panelmodule_components = self.panelmodule.components
        self.stains = self.panelmodule.stains
        self.channels = self.panelmodule.channels
        self.stain_marker_name = self.panelconfig.stain_marker_name
        self.markers = self.panelconfig.markers
        
    
    def init_params(self):
        """ Initialize parameters.
        """
        self.param_embedding = ParameterDict()
        for key in self.batch_keys:
            init_value = self.init_emb.clone()
            init_value[self.key_comp_batch.to_numpy()!=key] = 0.
            self.param_embedding[key] = ExtendableParameter((len(self.panelmodule.components), len(self.moe[key])), 1, init_value=init_value, allow_grad=True)

        self.param_embedding_mask = ParameterDict()
        if self.batch_order is None:
            for key in self.batch_keys:
                self.param_embedding_mask[key] = Parameter(torch.tensor(self.key_comp_batch.to_numpy()==key, dtype=torch.float32)[:,None], requires_grad=False)
        else:
            combmask = torch.full((len(self.key_comp_batch),), False)
            for key in [b for b in self.batch_order if b in self.batch_keys]:
                self.param_embedding_mask[key] = Parameter((~combmask).to(torch.float32)[:,None], requires_grad=False)
                combmask = combmask | torch.tensor(self.key_comp_batch.to_numpy()==key)
    
    def anchor_batches(self, anchor_bidx):
        """ Anchor batches to be not trainable.
            E.g. {"a":[0,1]} disables training for all parameters associated with indices [0,1] in batch "a".
        """
        for key in self.batch_keys:
            if key in anchor_bidx:
                self.param_embedding[key].freeze_selective((slice(None), list(anchor_bidx[key])))
    
    def export(self):
        """ Export self to dict.
        """
        exportdct = {}
        exportdct["panelconfig"] = self.panelconfig.export()
        exportdct["moe"] = self.moe.export()
        exportdct["key_comp_batch"] = self.key_comp_batch
        for key in ["key_comp_batch", "batch_order", "batch_keys", "max_abs_embedding", "init_emb"]:
            exportdct[key] = getattr(self, key)
        exportdct["state_dict"] = self.state_dict()
        exportdct["class"] = "embedding"
        
        return pdsave_expand_dict(exportdct)
    
    @classmethod
    def from_exported(cls, exported):
        """ Restore from exported.
        """
        exported = pdsave_collect_dict(exported)
        pem = cls()
        pem.panelconfig = PanelConfiguration.from_saved(exported=exported["panelconfig"])
        pem.panelmodule = PanelModule.from_panelconfig(pem.panelconfig)
        pem.init_add_from_panelmoduleconfig()
        pem.moe = MultiOrdinalEncoder.from_exported(exported["moe"])
        for key in ["key_comp_batch", "batch_order", "batch_keys", "max_abs_embedding", "init_emb"]:
            setattr(pem, key, exported[key])
        pem.init_params()
        pem.load_state_dict(exported["state_dict"])
        
        return pem
    
    def requires_grad_(self, requires_grad=True):
        """ Set requires_grad properly on all parameters.
        """
        for key in self.param_embedding:
            self.param_embedding[key].requires_grad_(requires_grad)
        for key in self.param_embedding_mask:
            self.param_embedding_mask[key].requires_grad_(False)
    
    def get_embedding_part(self, key, masked=True):
        """ Get part of embedding for specific batch key.
            Clips to max_abs_embedding, keeping gradients, if enables.
        """
        embedding = self.param_embedding[key]()
        if self.max_abs_embedding is not None:
            embedding = diffclamp(embedding, -self.max_abs_embedding, self.max_abs_embedding)
        if masked:
            embedding = self.param_embedding_mask[key] * embedding
        return embedding
    
    def forward(self, idx_batch, mask_unused=False):
        """ Get embedding for batch indices.
            
            :param idx_batch: dict. Batch index for every relevant batch key.
            
            :param mask_unused: bool. Whether to mask unused components, usually not necessary as panelmodule applies this automatically but sometimes useful.
        """
        embedding = sum([self.get_embedding_part(key)[:,idx_batch[key]] for key in self.batch_keys])
        if self.max_abs_embedding is not None: # clip again here if multiple parts
            embedding = diffclamp(embedding, -self.max_abs_embedding, self.max_abs_embedding)
        if mask_unused:
            embedding = embedding * self.panelmodule.param_components_mask[:,None]
        return embedding
    
    def freeze(self):
        for key in self.param_embedding:
            self.param_embedding[key].freeze()
    
    def extend(self, moe, extend_inds):
        """ Extend model by new batches.
            All new batches are trainable by default, use .anchor_batches is this should be modified.
            
            :param moe: MultiOrdinalEncoder. Extended batch encoder.
            
            :param extend_inds: dict. Extended indices dict, outout of moe.extend.
        """
        self.moe = moe
        for key in self.param_embedding:
            if key in extend_inds:
                self.param_embedding[key].extend(extend_inds[key])
    
    def get_spectra(self, idx_batch, mask_unused=True):
        """ Get spectra for batch index.
        """
        return self.panelmodule(self(idx_batch), mask_unused=mask_unused)
    
    def get_spectra_inv(self, idx_batch):
        """ Get inverted spectra for batch index.
        """
        return torch_invert_spectra(self.get_spectra(idx_batch))
    
    def get_unmixed(self, x, idx_batch):
        """ Get unmixed data for data x and batch index.
            Either by explicit inversion of by solving a least-squares problem.
            Explicit inversion is usually faster, probably stable enough.
            
            Bug in torch lstsq inversion? Just disable for now.
        """
        lstsq = False
        if lstsq:
            return torch_apply_unmixing_lstsq(self.get_spectra(idx_batch), x)
        else:
            return torch_apply_unmixing_inv(self.get_spectra_inv(idx_batch), x)
    
    def get_freeparam_unmix(self, idx_batch):
        """ For given idx_batch, returns both embedding as gradient accumulating parameter,
            as well as function to do the unmixing that doesn't mask any part of the embedding.
        """
        embedding = self(idx_batch)
        embedding = (embedding * self.panelmodule.param_components_mask.view([-1] + [1]*(embedding.dim()-1))).detach()
        embedding.requires_grad = True
        embedding.grad = torch.zeros_like(embedding)
        
        def fct_unmix(x):
            spectra = self.panelmodule(embedding, mask_unused=False)
            return torch_apply_unmixing_inv(torch_invert_spectra(spectra), x)
        
        return {"embedding":embedding}, fct_unmix

    def get_L1(self):
        """ Get L1 norm of all (used) embeddings, normalized to number of batches, summed across components.
        """
        get_mask = lambda key: self.panelmodule.apply_component_mask(self.param_embedding_mask[key])
        return torch.cat([self.get_embedding_part(key)[get_mask(key)[:,0]!=0].abs().mean(-1) for key in self.batch_keys]).sum()
    
    def get_learned_embedding(self):
        """ Get learned embedding, split into batch keys. Only to check overall convergence.
        """
        def get_single(key):
            mask = (self.param_embedding_mask[key].detach().cpu().numpy()[:,0]!=0) & (self.panelmodule.param_components_mask.detach().cpu().numpy()!=0)
            return pd.DataFrame(self.get_embedding_part(key).detach().cpu().numpy()[mask],
                                index=self.key_comp_batch.index[mask], columns=self.moe.labels[key])
        return {key: get_single(key) for key in self.batch_keys}


class FixedPanelModule(Module):
    """ Module that only uses a fixed panel for the unmixing.
    """
    TRAINED_EMBEDDING = False
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_data(cls, panel_spectra):
        """ 
            :param panel_spectra: pd.DataFrame. DataFrame containing the dye spectra, where the dyes are the index, and columns the fluorescence channels.
        """
        pem = cls()
        pem.panel_spectra = panel_spectra
        pem.panelmodule_components = []
        pem.stains = panel_spectra.index
        pem.stain_marker_name = panel_spectra.index
        pem.markers = panel_spectra.index
        pem.channels = panel_spectra.columns
        pem.batch_keys = []

        pem.param_spectra = Parameter(torch.tensor(panel_spectra.to_numpy(), dtype=torch.float32), requires_grad=False)
        
        return pem
    
    def export(self):
        """ Export self to dict.
        """
        exportdct = {}
        exportdct["panel_spectra"] = self.panel_spectra
        exportdct["class"] = "fixed"
        
        return pdsave_expand_dict(exportdct)
    
    @classmethod
    def from_exported(cls, exported):
        """ Restore from exported.
        """
        exported = pdsave_collect_dict(exported)
        pem = cls.from_data(exported["panel_spectra"])
        return pem
    
    def requires_grad_(self, requires_grad=True):
        """ 
        """
        pem.param_spectra.requires_grad_(False)
    
    def anchor_batches(self, anchor_bidx):
        """ Only here for compatibility reasons.
        """
        pass
    
    def forward(self, idx_batch, mask_unused=False):
        """ Only here for compatibility reasons.
        """
        return torch.tensor([])
    
    def freeze(self):
        """ Only here for compatibility reasons.
        """
        pass
    
    def extend(self, moe, extend_inds):
        """ Only here for compatibility reasons.
        """
        #self.moe = moe
        pass
    
    def get_spectra(self, idx_batch, mask_unused=True):
        """ Get spectra for batch index.
        """
        return self.param_spectra
    
    def get_spectra_inv(self, idx_batch):
        """ Get inverted spectra for batch index.
        """
        return torch_invert_spectra(self.get_spectra(idx_batch))
    
    def get_unmixed(self, x, idx_batch):
        """ Get unmixed data for data x and batch index.
        """
        return torch_apply_unmixing_inv(self.get_spectra_inv(idx_batch), x)
    
    def get_freeparam_unmix(self, idx_batch):
        raise NotImplementedError

    def get_L1(self):
        """ Only here for compatibility reasons.
        """
        return torch.tensor(0.)
    
    def get_learned_embedding(self):
        """ Only here for compatibility reasons.
        """
        return {}

