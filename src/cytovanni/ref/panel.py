"""Panel spectra from the PCA embedding of the dye spectra, and differentiable unmixing with them.

A panel embedding is one value per PCA component of all dyes of a panel (`PanelConfiguration`); every component belongs
to one dye. `PanelModule` turns an embedding into spectra of shape (stain, channel) and computes unmixing errors and
Jacobians with respect to the embedding. `PanelEmbeddingModule` holds a trainable embedding per batch for the reference
model; `FixedPanelModule` offers the same interface with one fixed spectra matrix.

Embeddings are in units of the standard deviation of the measured embeddings of each component (the PCA embeddings are
normalised to mean 0 and standard deviation 1 per component), so an embedding step of 1 is about the typical spread of
the measured spectra.
"""
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
    """Spectra of a panel as a function of its panel embedding; no trainable parameters.

    Holds the PCA model of the dye spectra from a `PanelConfiguration`: the offset (mean spectrum) per stain and the
    principal components, each belonging to one stain. The spectra for an embedding ``e`` are

        ``S[stain, channel] = offset[stain, channel] + sum_k components[stain, channel, k] * e[k]``,

    clipped to be non-negative. Components that are not used in the panel (``panelconfig.pca_components_use``) are
    masked by default, but the embedding still has one entry per component of the panel configuration, e.g. 130 entries
    if only 40 of 130 components are used.

    Create it with `from_panelconfig` or `from_exported`. Attributes: `channels`, `components` and `stains` (numpy
    arrays of names), `panelconfig`.

    The ``*_ashunmx`` and unmixing-error methods take intensities ``x`` of shape (..., channel) on the linear scale of
    the spectra, i.e. calibrated data (for a fitted `SpectralFitTransformer` additionally multiplied by its channel
    factors, ``adata.var['sfm_calibration_factor']``), and work on ``arcsinh(unmixed * stainscale / arcsinh_cofactor)``
    (stainscale 1 where not given). The ``*_fromad`` methods take an AnnData instead and, by default
    (``apply_factors=True``), apply the channel and stain factors that ``add_standardisation_parameters`` of a fitted
    `SpectralFitTransformer` stored in it. Results are on the device of the module.
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_panelconfig(cls, panelconfig):
        """Create the module from a panel configuration.

        Parameters
        ----------
        panelconfig : PanelConfiguration
            Panel with its PCA data (``pca_offset``, ``pca_components``, ``pca_components_stain``,
            ``pca_components_use``, set by ``PanelConfiguration.collect_PCA_data``).

        Returns
        -------
        PanelModule
        """
        pm = cls()
        pm.panelconfig = panelconfig
        
        pm.channels = pm.panelconfig.pca_components.columns.to_numpy()
        pm.components = pm.panelconfig.pca_components.index.to_numpy()
        pm.stains = pm.panelconfig.pca_offset.index.to_numpy()
        
        # general offset (stain, channel)
        pm.param_offset = Parameter(torch.tensor(pm.panelconfig.pca_offset.to_numpy(), dtype=torch.float32), requires_grad=False)
        
        # principal components (stain, channel, component)
        param_components = np.zeros((len(pm.stains), len(pm.channels), len(pm.components)))
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
        """Restore a module from the output of `export`.

        Parameters
        ----------
        exported : dict
            Dict with ``"panelconfig"`` (exported `PanelConfiguration`) and ``"state_dict"``.

        Returns
        -------
        PanelModule
        """
        panelconfig = PanelConfiguration.from_saved(exported=exported["panelconfig"])
        pm = cls.from_panelconfig(panelconfig)
        pm.load_state_dict(exported["state_dict"])
        
        return pm
    
    def export(self):
        """Export to a dict (``"panelconfig"``: ``PanelConfiguration.export()``, ``"state_dict"``) for saving."""
        exported = {}
        exported["panelconfig"] = self.panelconfig.export()
        exported["state_dict"] = self.state_dict()
        return exported
    
    def requires_grad_(self, requires_grad=True):
        """Keep all parameters non-trainable, whatever `requires_grad` is; returns self."""
        self.param_offset.requires_grad_(False)
        self.param_components.requires_grad_(False)
        self.param_components_mask.requires_grad_(False)
        return self
    
    def forward(self, x_embedding, clamp_differentiable=True, mask_unused=True):
        """Compute the spectra for one or many embeddings.

        Parameters
        ----------
        x_embedding : torch.Tensor of shape (component, ...)
            Panel embedding(s), component axis first; the remaining axes become the leading axes of the result, e.g.
            (component, events) for one embedding per event.
        clamp_differentiable : bool, default True
            Clip negative spectra entries to 0 with a straight-through gradient (`diffclamp`: the gradient also flows
            where the value was clipped); if False, a plain clamp (no gradient where clipped).
        mask_unused : bool, default True
            Set the embedding of components not used in the panel to 0 (also non-finite values there). If False, all
            components enter the spectra.

        Returns
        -------
        torch.Tensor of shape (..., stain, channel)
            Spectra, non-negative.
        """
        offset = self.param_offset.view( [1]*(len(x_embedding.shape)-1) + list(self.param_offset.shape) )
        if mask_unused:
            # zero the unused components' embedding as well, so a non-finite unused entry cannot give NaN (NaN*0 = NaN)
            mask = self.param_components_mask.reshape([len(self.param_components_mask)]+[1]*(x_embedding.ndim-1)) != 0
            x_embedding = torch.where(mask, x_embedding, torch.zeros_like(x_embedding))
        comp = torch.einsum("ijk, k... -> ...ij", self.param_components * (self.param_components_mask if mask_unused else 1.), x_embedding)
        if clamp_differentiable:
            return diffclamp(offset + comp, 0, None)
        else:
            return (offset + comp).clamp(0, None)
    
    def apply_component_mask(self, emb):
        """Multiply `emb` of shape (component, ...) by the mask of used components (0 for unused ones)."""
        mask = self.param_components_mask.reshape([len(self.param_components_mask)]+[1]*(emb.ndim-1))
        return mask * emb

    def get_jacobian_ashunmx_jacrev(self, x, embedding, include_unused=True, arcsinh_cofactor=1500):
        """Jacobian of the unmixed, arcsinh-transformed data with respect to the embedding, via ``torch.func.jacrev``.

        Reference implementation with a very large memory footprint, only to check `get_jacobian_ashunmx` (same result,
        without the stain scaling).

        Parameters
        ----------
        x : torch.Tensor of shape (..., channel)
            Intensities on the linear scale of the spectra (see the class description).
        embedding : torch.Tensor of shape (component,)
            Panel embedding.
        include_unused : bool, default True
            Let the components not used in the panel enter the spectra, so the Jacobian is also non-zero for them. If
            False they are masked and their columns are 0.
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of `x`.

        Returns
        -------
        torch.Tensor of shape (..., stain, component)
            ``d arcsinh(unmixed / arcsinh_cofactor) / d embedding``.
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
        """Jacobian of the unmixed, arcsinh-transformed data with respect to the embedding.

        Computes the Jacobian of the inverted spectra once (with ``jacrev``) and applies the chain rule per event, so
        the memory use is much lower than with `get_jacobian_ashunmx_jacrev`.

        Parameters
        ----------
        x : torch.Tensor of shape (..., channel)
            Intensities on the linear scale of the spectra (see the class description).
        embedding : torch.Tensor of shape (component,)
            Panel embedding.
        stainscale : torch.Tensor of shape (..., stain), optional
            Linear factor per stain applied after unmixing (e.g. the stain factors of a fitted transformer). Default: 1.
        include_unused : bool, default True
            Let the components not used in the panel enter the spectra, so the Jacobian is also non-zero for them. If
            False they are masked and their columns are 0.
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of `x`.

        Returns
        -------
        torch.Tensor of shape (..., stain, component)
            ``d arcsinh(unmixed * stainscale / arcsinh_cofactor) / d embedding``.
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
        """Unmix with the spectra of an embedding, scale per stain and arcsinh-transform.

        Differentiable with respect to the embedding (spectra clipped at 0 with a plain clamp).

        Parameters
        ----------
        x : torch.Tensor of shape (..., channel)
            Intensities on the linear scale of the spectra (see the class description).
        embedding : torch.Tensor of shape (component, ...)
            Panel embedding, component axis first (as in `forward`); its remaining axes broadcast with the leading axes
            of `x`, e.g. (component,) for one embedding for all events, or (component, events) for one per event.
        stainscale : torch.Tensor of shape (..., stain), optional
            Linear factor per stain applied after unmixing. Default: 1.
        include_unused : bool, default True
            Let the components not used in the panel enter the spectra; if False they are masked.
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of `x`.

        Returns
        -------
        torch.Tensor of shape (..., stain)
            ``arcsinh(unmixed * stainscale / arcsinh_cofactor)``.
        """
        x, embedding, stainscale = x.to(self.device), embedding.to(self.device), stainscale.to(self.device) if stainscale is not None else 1.
        inv_spectra = torch_invert_spectra(self(embedding, clamp_differentiable=False, mask_unused=not include_unused))
        return (torch_apply_unmixing_inv(inv_spectra, x) * stainscale / arcsinh_cofactor).arcsinh()
    
    def get_unmixing_resistance(self, x_np, embedding_np, stainscale_np=None, arcsinh_cofactor=1500, embedding_step=1., include_unused=False, single_step=False, gradstep_maxnorm=False, batch_size=1e4):
        """Estimate per event and stain how strongly the unmixed value is affected by spectrum errors.

        For every event and stain, the unmixed value ``val = arcsinh(unmixed * stainscale / arcsinh_cofactor)`` is
        recomputed with the embedding changed by plus and minus a step, and the error ``val_std`` is the mean absolute
        change of the two. The step is either along the component with the largest absolute Jacobian for that event and
        stain (`single_step`), or along the Jacobian direction itself (scaled as set by `gradstep_maxnorm`). The
        resistance ``R = arcsinh(|val| / val_std / 20)`` compares the value with this error on an arcsinh scale: large R
        means the value is large compared to the change a plausible spectrum error would cause.

        Runs on the device of the module (slow on the CPU), in chunks of `batch_size` events.

        Parameters
        ----------
        x_np : numpy.ndarray of shape (events, channel)
            Intensities on the linear scale of the spectra (see the class description).
        embedding_np : numpy.ndarray of shape (component,)
            Panel embedding, e.g. ``adata.uns['panel_embedding']``.
        stainscale_np : numpy.ndarray of shape (stain,), optional
            Linear factor per stain applied after unmixing, e.g. ``adata.varm['sfm_stain_factor'].iloc[0]``. Default: 1.
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of the data.
        embedding_step : float, default 1.0
            Size of the embedding step, in standard deviations of the measured embeddings.
        include_unused : bool, default False
            Let the components not used in the panel enter the spectra, so steps along them change the result; if False
            they are masked.
        single_step : bool, default False
            If True, step only along the component with the largest absolute Jacobian (per event and stain), by
            +-`embedding_step`. If False, step along the Jacobian direction, normalised as set by `gradstep_maxnorm`.
        gradstep_maxnorm : bool, default False
            Only if not `single_step`: if True, the largest component change of the step is `embedding_step`; if False,
            the absolute changes of all components sum to `embedding_step`.
        batch_size : int or float, default 1e4
            Events per chunk, to limit memory use (larger chunks hardly help).

        Returns
        -------
        val : numpy.ndarray of shape (events, stain)
            Unmixed, scaled, arcsinh-transformed values.
        val_std : numpy.ndarray of shape (events, stain)
            Error estimate: mean absolute change of `val` under the embedding step. Where the Jacobian of a stain is
            zero (e.g. events with all-zero intensities) the gradient step has no direction and the error is 0.
        R : numpy.ndarray of shape (events, stain)
            ``arcsinh(|val| / val_std / 20)``; 0 where ``val == 0`` (no signal), inf where ``val != 0`` but the error is
            0.

        See Also
        --------
        get_unmixing_resistance_fromad : R directly from an AnnData.
        get_compmean_errorbars : the same idea for population means.
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
            embedding_mod[ind_maxcomp.flatten(), arN.flatten(), arNstain.flatten(),:] += (torch.ones((Nstain*N,2)) * torch.tensor([[embedding_step,-embedding_step]], dtype=torch.float32)).to(self.device)
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
                norm = jacob.abs().max(-1, keepdims=True).values
            else:
                norm = jacob.abs().sum(-1, keepdims=True)
            # zero Jacobian: no direction, zero step (instead of 0/0)
            direction = torch.where(norm>0, jacob / norm, torch.zeros_like(jacob))
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
        # no signal, no resistance: R = 0 where val == 0 (also where val_std == 0, instead of 0/0)
        R = np.arcsinh(np.divide(np.abs(val), val_std, out=np.zeros_like(val), where=val!=0)/20.)

        return val, val_std, R

    def _get_x_stainscale_fromad(self, adata, key_layer, apply_factors=True):
        """Intensities of `adata` (events, channel) and stain factors (stain,) or None, for the ``*_fromad`` methods.

        With `apply_factors`, uses the parameters that ``add_standardisation_parameters`` of a fitted
        `SpectralFitTransformer` stored in `adata`, where present: the layer is multiplied by
        ``adata.var['sfm_calibration_factor']`` and the stain factors are ``adata.varm['sfm_stain_factor']``.
        """
        x_np = adata[:,self.channels].layers[key_layer]
        stainscale_np = None
        if apply_factors:
            if "sfm_calibration_factor" in adata.var:
                channelscale = adata.var.loc[list(self.channels), "sfm_calibration_factor"].to_numpy(dtype=float)
                if np.isnan(channelscale).any():
                    raise ValueError("adata.var['sfm_calibration_factor'] is NaN for some channels of the panel!")
                x_np = x_np * channelscale[None]
            if "sfm_stain_factor" in adata.varm:
                stainscale_np = np.asarray(adata.varm["sfm_stain_factor"])[0].astype(float)
                if stainscale_np.shape[0]!=len(self.stains):
                    raise ValueError(f"adata.varm['sfm_stain_factor'] has {stainscale_np.shape[0]} stains, "
                                     f"the panel {len(self.stains)}!")
        return x_np, stainscale_np

    def get_unmixing_resistance_fromad(self, adata, embedding_np, key_layer="calibrated", arcsinh_cofactor=1500, embedding_step=1., include_unused=False, single_step=False, gradstep_maxnorm=False, batch_size=1e4, label_marker=True, apply_factors=True):
        """Unmixing resistance R (see `get_unmixing_resistance`) for the events of an AnnData.

        With `apply_factors`, the data are put on the scale of the fitted model first, using the parameters that
        ``add_standardisation_parameters`` of a fitted `SpectralFitTransformer` stored in `adata` (where present):
        the layer is multiplied by the channel factors ``adata.var['sfm_calibration_factor']`` and the unmixed values
        by the stain factors ``adata.varm['sfm_stain_factor']``.

        Parameters
        ----------
        adata : anndata.AnnData
            Measurement with all `channels` of the panel.
        embedding_np : numpy.ndarray of shape (component,)
            Panel embedding.
        key_layer : str, default "calibrated"
            Layer with the intensities to unmix (linear scale of the spectra).
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of the data.
        embedding_step : float, default 1.0
            Size of the embedding step, in standard deviations of the measured embeddings.
        include_unused : bool, default False
            Let the components not used in the panel enter the spectra.
        single_step : bool, default False
            Step along the single most important component (True) or along the Jacobian direction (False).
        gradstep_maxnorm : bool, default False
            Only if not `single_step`: normalise the largest component change (True) or the sum of absolute changes
            (False) to `embedding_step`.
        batch_size : int or float, default 1e4
            Events per chunk.
        label_marker : bool, default True
            Label the columns with ``panelconfig.stain_marker_name`` (``"stain | marker"``) instead of the stain names.
        apply_factors : bool, default True
            Apply the stored channel and stain factors (see above). If False, the layer is used as given.

        Returns
        -------
        pandas.DataFrame
            R, index ``adata.obs.index``, one column per stain.

        Raises
        ------
        ValueError
            If the stored channel factors are NaN for a panel channel, or the stored stain factors do not have one
            entry per stain.
        """
        x_np, stainscale_np = self._get_x_stainscale_fromad(adata, key_layer, apply_factors=apply_factors)
        R = self.get_unmixing_resistance(x_np, embedding_np, stainscale_np=stainscale_np, arcsinh_cofactor=arcsinh_cofactor, embedding_step=embedding_step,
                                         include_unused=include_unused, single_step=single_step, gradstep_maxnorm=gradstep_maxnorm,
                                         batch_size=batch_size)[-1]
        return pd.DataFrame(R, index=adata.obs.index, columns=self.panelconfig.stain_marker_name if label_marker else self.stains)

    def get_grouped_jac_fromad(self, adata, embedding_np, stainscale_np=None, key_layer="calibrated", arcsinh_cofactor=1500, batch_size=1e4, label_marker=True, apply_factors=True):
        """Mean Jacobian of the unmixed, arcsinh-transformed data with respect to the embedding, over an AnnData.

        Shows which panel components (also those not used in the panel: all components enter, ``include_unused=True``)
        most affect the unmixed values of a population; subset `adata` to the events of interest first.

        Parameters
        ----------
        adata : anndata.AnnData
            Measurement (subset) with all `channels` of the panel.
        embedding_np : numpy.ndarray of shape (component,)
            Panel embedding at which the Jacobian is evaluated.
        stainscale_np : numpy.ndarray of shape (stain,), optional
            Linear factor per stain applied after unmixing. Default: the stored stain factors with `apply_factors`
            (see `get_unmixing_resistance_fromad`), else 1.
        key_layer : str, default "calibrated"
            Layer with the intensities to unmix (linear scale of the spectra).
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of the data.
        batch_size : int or float, default 1e4
            Events per chunk.
        label_marker : bool, default True
            Label the columns with ``panelconfig.stain_marker_name`` instead of the stain names.
        apply_factors : bool, default True
            Multiply the layer by the stored channel factors and use the stored stain factors, as in
            `get_unmixing_resistance_fromad`. If False, the layer is used as given.

        Returns
        -------
        pandas.DataFrame
            Mean of ``d arcsinh(unmixed * stainscale / arcsinh_cofactor) / d embedding``; index: components, columns:
            stains.
        """
        x_np, stainscale_stored = self._get_x_stainscale_fromad(adata, key_layer, apply_factors=apply_factors)
        if stainscale_np is None:
            stainscale_np = stainscale_stored
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
        """Rank the panel components by their mean effect on one stain, for an AnnData with standardisation parameters.

        Runs `get_grouped_jac_fromad` with the embedding and stain factors that ``add_standardisation_parameters`` of a
        fitted `SpectralFitTransformer` stored in `adata` (``adata.uns['panel_embedding']``,
        ``adata.varm['sfm_stain_factor']``, and the channel factors ``adata.var['sfm_calibration_factor']``) and returns
        the components sorted by the absolute mean Jacobian for `stain`, marking whether each is used in the panel.

        Parameters
        ----------
        adata : anndata.AnnData
            Measurement (subset to the events of interest) with the standardisation parameters added.
        stain : str
            Stain to look at, as in ``panelconfig.stain_marker_name`` (e.g. ``"APC | CD33"``).
        key_layer : str, default "calibrated"
            Layer with the intensities to unmix.
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of the data.
        s_dig : int, default 3
            Number of decimal places the Jacobian is rounded to.
        include : int or None, default 10
            Keep only this many components with the largest absolute Jacobian (None: all).

        Returns
        -------
        df : pandas.DataFrame
            Index: components; columns ``"abs. Jacobian"``, ``"Jacobian"`` and ``"in Panel"`` (bool), sorted by the
            absolute Jacobian, descending.
        fct_style : callable
            ``fct_style(df)`` returns a pandas Styler that colours components in the panel green and the others orange.

        Examples
        --------
        >>> df, fct_style = pm.get_grouped_jac_single_fromad(adata[gate.apply(adata)], "APC | CD33")
        >>> fct_style(df)
        """
        jac = self.get_grouped_jac_fromad(adata, embedding_np=adata.uns["panel_embedding"], key_layer=key_layer,
                                          arcsinh_cofactor=arcsinh_cofactor, batch_size=1e4, label_marker=True, apply_factors=True)
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
        """Spectra for an embedding as a DataFrame.

        Parameters
        ----------
        embedding : array-like of shape (component,)
            Panel embedding.
        label_marker : bool, default True
            Label the rows with ``panelconfig.stain_marker_name`` instead of the stain names.
        include_unused : bool, default False
            Let the components not used in the panel enter the spectra.

        Returns
        -------
        pandas.DataFrame
            Spectra (stain, channel), clipped at 0; index: stains, columns: `channels`.
        """
        with torch.no_grad():
            spectra = pd.DataFrame(self(torch.tensor(embedding, dtype=torch.float32).to(self.device), mask_unused=not include_unused).detach().cpu().numpy(),
                                 index=self.panelconfig.stain_marker_name if label_marker else self.stains, columns=self.channels)
        return spectra
    
    def get_compmean_errorbars(self, x_np, embedding_np, arcsinh_cofactor=1500, embedding_step=1., label_marker=True, use_median=False, include_unused=False, stainscale_np=None):
        """Estimate how strongly spectrum errors affect the mean unmixed value of a population, per stain.

        The mean (or median) over the events of ``arcsinh(unmixed * stainscale / arcsinh_cofactor)`` is computed per
        stain; for every stain, the component with the largest absolute Jacobian of that mean is changed by
        +-`embedding_step` and the error is the mean absolute change of the mean.

        The Jacobian is computed for all events at once with ``jacrev``: for 110 components and 23 stains this needs
        about 1 GB of GPU memory per 100 events, so pass a subsample.

        Parameters
        ----------
        x_np : numpy.ndarray of shape (events, channel)
            Intensities of the population on the linear scale of the spectra.
        embedding_np : numpy.ndarray of shape (component,)
            Panel embedding.
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of the data.
        embedding_step : float, default 1.0
            Size of the embedding step, in standard deviations of the measured embeddings.
        label_marker : bool, default True
            Label the rows with ``panelconfig.stain_marker_name`` instead of the stain names.
        use_median : bool, default False
            Use the median instead of the mean (the output column is still called ``"mean"``).
        include_unused : bool, default False
            Let the components not used in the panel enter the spectra.
        stainscale_np : numpy.ndarray of shape (stain,), optional
            Linear factor per stain applied after unmixing. Default: 1.

        Returns
        -------
        pandas.DataFrame
            Index: stains; columns ``"mean"`` (arcsinh scale), ``"std"`` (the error estimate) and ``"z"`` (mean / std).

        See Also
        --------
        get_clustermeans_withz : the same for every cluster of an AnnData.
        """
        with torch.no_grad():
            # turn into tensors
            x = torch.as_tensor(x_np, dtype=torch.float32).to(self.device)
            embedding = torch.as_tensor(embedding_np, dtype=torch.float32).to(self.device)
            stainscale = torch.as_tensor(stainscale_np, dtype=torch.float32).to(self.device) if stainscale_np is not None else 1.
            # function to get cluster means of unmixed
            fct_reduce = lambda x: x.mean(-2) if not use_median else x.median(-2).values
            fct_mean = lambda x, emb: fct_reduce((torch_apply_unmixing_inv(torch_invert_spectra(
                                                    self(emb, clamp_differentiable=False, mask_unused=not include_unused)
                                                    ), x) * stainscale / arcsinh_cofactor).arcsinh())
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
    
    def get_clustermeans_withz(self, embedding, adata, key_cluster, key_layer="calibrated", cluster_reorder=True, Nmax=4000, arcsinh_cofactor=1500, embedding_step=1., label_marker=True, include_unused=True, apply_factors=True):
        """Mean unmixed values per cluster, with z-scores that take possible unmixing errors into account.

        For every cluster, up to `Nmax` random events are passed to `get_compmean_errorbars`; ``z = mean / std``
        indicates how clearly the mean differs from 0 given the possible spectrum errors.

        Parameters
        ----------
        embedding : numpy.ndarray of shape (component,)
            Panel embedding.
        adata : anndata.AnnData
            Measurement with all `channels` of the panel and the cluster labels.
        key_cluster : str
            Column of ``adata.obs`` with the cluster labels.
        key_layer : str, default "calibrated"
            Layer with the intensities to unmix (linear scale of the spectra).
        cluster_reorder : bool, default True
            Reorder rows and columns of the results by the hierarchical clustering of the means (``seaborn.clustermap``;
            the figure is closed).
        Nmax : int, default 4000
            Maximum number of events per cluster (random subsample, numpy's global random state).
        arcsinh_cofactor : float, default 1500
            Cofactor of the arcsinh transform, in the units of the data.
        embedding_step : float, default 1.0
            Size of the embedding step, in standard deviations of the measured embeddings.
        label_marker : bool, default True
            Label the columns with ``panelconfig.stain_marker_name`` instead of the stain names.
        include_unused : bool, default True
            Let the components not used in the panel enter the spectra (note: the default differs from
            `get_compmean_errorbars`).
        apply_factors : bool, default True
            Multiply the layer by the stored channel factors and apply the stored stain factors, as in
            `get_unmixing_resistance_fromad`. If False, the layer is used as given.

        Returns
        -------
        means : pandas.DataFrame
            Mean arcsinh-transformed unmixed value; index: clusters, columns: stains.
        zs : pandas.DataFrame
            z-scores, same layout.
        """
        clusters = np.unique(adata.obs[key_cluster])
        x_all, stainscale_np = self._get_x_stainscale_fromad(adata, key_layer, apply_factors=apply_factors)
        outs = []
        for c in clusters:
            X = x_all[adata.obs[key_cluster].to_numpy()==c]
            X = X[np.random.permutation(X.shape[0])[:Nmax]]
            outs.append(self.get_compmean_errorbars(X, embedding,
                                                    arcsinh_cofactor=arcsinh_cofactor, embedding_step=embedding_step,
                                                    label_marker=label_marker, include_unused=include_unused,
                                                    stainscale_np=stainscale_np))
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
        """Device of the module's parameters."""
        return self.param_components_mask.device

class PanelEmbeddingModule(Module):
    """Trainable panel embedding per batch, turned into spectra by a `PanelModule` (reference model).

    Every panel component is assigned to one batch key (`key_comp_batch`), e.g. most components to the measurement day
    and the components of particularly unstable dyes to the individual batch. Each key has an embedding part of shape
    (component, batches of that key), masked to the components it may change; the embedding of an event is the sum of
    the parts of its batches, limited to +-`max_abs_embedding`. Parameters exist for all components, but the
    `PanelModule` masks the unused ones. Batches can be anchored (frozen) and added later (`extend`).

    Create it with `from_data` or `from_exported`. ``TRAINED_EMBEDDING = True`` tells the transformers that the spectra
    are fitted.
    """
    TRAINED_EMBEDDING = True
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_data(cls, panelconfig, moe, key_comp_batch, batch_order=None, anchor_bidx={}, fixed_init=None, init_emb=None, max_abs_embedding=6):
        """Create the module for a panel and a batch encoder.

        Parameters
        ----------
        panelconfig : PanelConfiguration
            Panel with its PCA data (see `PanelModule.from_panelconfig`).
        moe : MultiOrdinalEncoder
            Batch encoder containing all batch keys used in `key_comp_batch`.
        key_comp_batch : pandas.Series
            Batch key per component: index all component names of `panelconfig` (any order), values batch keys. E.g.
            ``pd.Series(["a", "b"], index=["PC0", "PC1"])`` fits the embedding of PC0 per batch of key "a" and that of
            PC1 per batch of key "b".
        batch_order : list of str, optional
            Order of the batch keys from coarse to fine, e.g. ``["batch", "batch_ref"]``. With an order, a key also gets
            parameters for the components assigned to all later keys (initialised at 0): for the example above with
            ``batch_order=["a", "b"]``, PC1 has a part per batch of "b" plus a part per batch of "a" shared by the "b"
            batches within it. Intended to speed up training when there are many "b" batches per "a" batch that should
            have more similar embeddings than batches of different "a". Must contain all keys of `key_comp_batch`.
            Default: every key only changes its own components.
        anchor_bidx : dict of str to list of int, default {}
            Batch indices per key whose embedding parts are frozen at their initial value, e.g. ``{"a": [0, 1]}``;
            usually from ``moe.transform_dict_simple(anchors)``.
        fixed_init : str, optional
            Name of a measurement in ``panelconfig.pca_embedding`` (e.g. a compensation run) whose embedding initialises
            all batches (missing entries 0). Ignored if `init_emb` is given. Default: all zeros.
        init_emb : array-like of shape (component,), optional
            Initial embedding of all batches. Each key part starts with the values of its own components (0 elsewhere),
            so the summed embedding starts at the initial embedding.
        max_abs_embedding : float or None, default 6
            Limit of the embedding of every key part and of the summed embedding, in standard deviations of the measured
            embeddings (keeps inferred embeddings in the plausible range). Straight-through (`diffclamp`): the forward
            value is clipped, the stored parameter is not and can exceed the limit. None: no limit.

        Returns
        -------
        PanelEmbeddingModule

        Raises
        ------
        OverlapStandardisationException
            If a batch key of `key_comp_batch` is missing from `moe` or from `batch_order`.
        KeyError
            If a component of the panel is missing from the index of `key_comp_batch`.

        Examples
        --------
        >>> key_comp_batch = pd.Series("batch", index=panelconfig.pca_components.index.to_numpy(), dtype=str)
        >>> key_comp_batch.loc[panelconfig.pca_components_stain == "BV786"] = "batch_ref"
        >>> pem = PanelEmbeddingModule.from_data(panelconfig, moe, key_comp_batch, batch_order=["batch", "batch_ref"],
        ...                                      anchor_bidx=moe.transform_dict_simple(anchors),
        ...                                      fixed_init="HD7_Compensation_20240418_tcell", max_abs_embedding=6)
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
        """Copy component, stain, channel and marker names from the panel module and config to self."""
        self.panelmodule_components = self.panelmodule.components
        self.stains = self.panelmodule.stains
        self.channels = self.panelmodule.channels
        self.stain_marker_name = self.panelconfig.stain_marker_name
        self.markers = self.panelconfig.markers
        
    
    def init_params(self):
        """Create the embedding part (component, batches) and the component mask of every batch key."""
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
        """Freeze the embedding parts of the given batches at their current value.

        Parameters
        ----------
        anchor_bidx : dict of str to list of int
            Batch indices per batch key, e.g. ``{"a": [0, 1]}`` freezes batches 0 and 1 of key "a". Keys that are not
            batch keys of the module are ignored.
        """
        for key in self.batch_keys:
            if key in anchor_bidx:
                self.param_embedding[key].freeze_selective((slice(None), list(anchor_bidx[key])))
    
    def export(self):
        """Export to a dict for saving (pandas objects expanded); restore with `from_exported`."""
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
        """Restore a module from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export`.

        Returns
        -------
        PanelEmbeddingModule
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
        """Enable or disable gradients of the embedding parts (masks never trainable); returns self."""
        for key in self.param_embedding:
            self.param_embedding[key].requires_grad_(requires_grad)
        for key in self.param_embedding_mask:
            self.param_embedding_mask[key].requires_grad_(False)
        return self
    
    def get_embedding_part(self, key, masked=True):
        """Embedding part of one batch key, limited to +-`max_abs_embedding` (straight-through).

        Parameters
        ----------
        key : str
            Batch key.
        masked : bool, default True
            Set the components this key may not change to 0.

        Returns
        -------
        torch.Tensor of shape (component, batches of `key`)
        """
        embedding = self.param_embedding[key]()
        if self.max_abs_embedding is not None:
            embedding = diffclamp(embedding, -self.max_abs_embedding, self.max_abs_embedding)
        if masked:
            embedding = self.param_embedding_mask[key] * embedding
        return embedding
    
    def forward(self, idx_batch, mask_unused=False):
        """Panel embedding per event: sum of the embedding parts of its batches, limited to +-`max_abs_embedding`.

        Parameters
        ----------
        idx_batch : dict of str to torch.Tensor
            Batch index tensor of shape (events,) per batch key (e.g. from the batch encoder).
        mask_unused : bool, default False
            Set components not used in the panel to 0; usually unnecessary, since the `PanelModule` masks them anyway.

        Returns
        -------
        torch.Tensor of shape (component, events)
        """
        embedding = sum([self.get_embedding_part(key)[:,idx_batch[key]] for key in self.batch_keys])
        if self.max_abs_embedding is not None: # clip again here if multiple parts
            embedding = diffclamp(embedding, -self.max_abs_embedding, self.max_abs_embedding)
        if mask_unused:
            embedding = embedding * self.panelmodule.param_components_mask[:,None]
        return embedding
    
    def freeze(self):
        """Freeze all embedding parts (all current batches)."""
        for key in self.param_embedding:
            self.param_embedding[key].freeze()
    
    def extend(self, moe, extend_inds):
        """Add new batches; their embedding parts start at the initial embedding and are trainable.

        The existing batches are frozen at their current value (`ExtendableParameter.extend`); use `anchor_batches` to
        freeze new batches as well.

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Extended batch encoder (replaces the stored one).
        extend_inds : dict of str to list of int
            New batch indices per batch key, the output of ``moe.extend``.
        """
        self.moe = moe
        for key in self.param_embedding:
            if key in extend_inds:
                self.param_embedding[key].extend(extend_inds[key])
    
    def get_spectra(self, idx_batch, mask_unused=True):
        """Spectra per event, shape (events, stain, channel), for the batch indices `idx_batch` (see `forward`).

        `mask_unused` (default True) is passed to the `PanelModule`.
        """
        return self.panelmodule(self(idx_batch), mask_unused=mask_unused)
    
    def get_spectra_inv(self, idx_batch):
        """Inverted spectra per event, shape (events, channel, stain), for the batch indices `idx_batch`."""
        return torch_invert_spectra(self.get_spectra(idx_batch))
    
    def get_unmixed(self, x, idx_batch):
        """Unmix intensities with the spectra of their batches.

        Always uses the explicit inverse (`torch_invert_spectra`); the least-squares variant in the code is disabled.

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Intensities on the linear scale of the spectra.
        idx_batch : dict of str to torch.Tensor
            Batch index tensor of shape (events,) per batch key.

        Returns
        -------
        torch.Tensor of shape (events, stain)
        """
        lstsq = False
        if lstsq:
            return torch_apply_unmixing_lstsq(self.get_spectra(idx_batch), x)
        else:
            return torch_apply_unmixing_inv(self.get_spectra_inv(idx_batch), x)
    
    def get_freeparam_unmix(self, idx_batch):
        """Embedding as a free leaf tensor plus an unmixing function, to get gradients with respect to the embedding.

        The embedding for `idx_batch` is computed, unused components are set to 0, and the result is detached as a new
        leaf tensor with ``requires_grad`` and a zero gradient. The returned function unmixes with spectra from this
        tensor without masking, so after a backward pass ``params["embedding"].grad`` holds the gradient for every
        component, including those not used in the panel.

        Parameters
        ----------
        idx_batch : dict of str to torch.Tensor
            Batch index tensor of shape (events,) per batch key.

        Returns
        -------
        params : dict
            ``{"embedding": torch.Tensor of shape (component, events)}``.
        fct_unmix : callable
            ``fct_unmix(x)`` with `x` of shape (events, channel) returns the unmixed data (events, stain).
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
        """L1 penalty of the embedding: per batch key and component, mean absolute value over the batches, summed.

        Only components that are used in the panel and that the key may change enter; the limited (clipped) values are
        used. Returns a 0-dim tensor.
        """
        get_mask = lambda key: self.panelmodule.apply_component_mask(self.param_embedding_mask[key])
        return torch.cat([self.get_embedding_part(key)[get_mask(key)[:,0]!=0].abs().mean(-1) for key in self.batch_keys]).sum()
    
    def get_learned_embedding(self):
        """Current embedding parts, to check convergence.

        Returns
        -------
        dict of str to pandas.DataFrame
            Per batch key: index the used components the key may change, columns the batch labels (``moe.labels[key]``);
            values limited to +-`max_abs_embedding`.
        """
        def get_single(key):
            mask = (self.param_embedding_mask[key].detach().cpu().numpy()[:,0]!=0) & (self.panelmodule.param_components_mask.detach().cpu().numpy()!=0)
            return pd.DataFrame(self.get_embedding_part(key).detach().cpu().numpy()[mask],
                                index=self.key_comp_batch.index[mask], columns=self.moe.labels[key])
        return {key: get_single(key) for key in self.batch_keys}


class FixedPanelModule(Module):
    """One fixed spectra matrix for all batches, with the interface of `PanelEmbeddingModule`.

    For models that only fit scale factors. ``TRAINED_EMBEDDING = False``; the embedding-related methods are no-ops.
    Create it with `from_data` or `from_exported`.
    """
    TRAINED_EMBEDDING = False
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_data(cls, panel_spectra):
        """Create the module from a spectra DataFrame.

        Parameters
        ----------
        panel_spectra : pandas.DataFrame
            Spectra (stain, channel): index the stains (also used as `stain_marker_name` and `markers`), columns the
            fluorescence channels; on the linear scale of the data to unmix (e.g. calibrated spectra for calibrated
            data).

        Returns
        -------
        FixedPanelModule
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
        """Export to a dict for saving (spectra expanded with `pdsave_expand_dict`); restore with `from_exported`."""
        exportdct = {}
        exportdct["panel_spectra"] = self.panel_spectra
        exportdct["class"] = "fixed"
        
        return pdsave_expand_dict(exportdct)
    
    @classmethod
    def from_exported(cls, exported):
        """Restore a module from the output of `export`."""
        exported = pdsave_collect_dict(exported)
        pem = cls.from_data(exported["panel_spectra"])
        return pem
    
    def requires_grad_(self, requires_grad=True):
        """Keep the spectra non-trainable, whatever `requires_grad` is; returns self."""
        self.param_spectra.requires_grad_(False)
        return self
    
    def anchor_batches(self, anchor_bidx):
        """Do nothing (interface compatibility with `PanelEmbeddingModule`)."""
        pass
    
    def forward(self, idx_batch, mask_unused=False):
        """Return an empty tensor (no embedding; interface compatibility)."""
        return torch.tensor([])
    
    def freeze(self):
        """Do nothing (interface compatibility)."""
        pass
    
    def extend(self, moe, extend_inds):
        """Do nothing (interface compatibility; the batch encoder is not stored)."""
        #self.moe = moe
        pass
    
    def get_spectra(self, idx_batch, mask_unused=True):
        """Return the fixed spectra, shape (stain, channel), the same for every batch (no event axis)."""
        return self.param_spectra
    
    def get_spectra_inv(self, idx_batch):
        """Return the inverted fixed spectra, shape (channel, stain)."""
        return torch_invert_spectra(self.get_spectra(idx_batch))
    
    def get_unmixed(self, x, idx_batch):
        """Unmix `x` of shape (events, channel) with the fixed spectra; returns (events, stain)."""
        return torch_apply_unmixing_inv(self.get_spectra_inv(idx_batch), x)
    
    def get_freeparam_unmix(self, idx_batch):
        """Not available for fixed spectra (raises NotImplementedError)."""
        raise NotImplementedError

    def get_L1(self):
        """Return 0 (no embedding; interface compatibility)."""
        return torch.tensor(0.)
    
    def get_learned_embedding(self):
        """Return an empty dict (no embedding; interface compatibility)."""
        return {}

