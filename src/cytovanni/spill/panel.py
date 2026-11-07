"""Panel configuration: the dyes of a panel, their markers and the PCA components of their spectra used in the fit."""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict

class PanelConfiguration():
    """Dyes of a panel with their markers, and the PCA description of their spectra used as the panel embedding.

    Every dye ("stain") of the panel is described by the mean of its measured spectra and a chosen subset of its
    principal components (`SpilloverCollection.add_PCA_fit`); the embeddings of all chosen components together form
    the panel embedding that `cytovanni.ref.PanelEmbeddingModule` fits per batch. Create it with `from_csv`, then add
    the PCA data with `collect_PCA_data`; save and load with `save` / `from_saved`. The constructor itself creates an
    empty object.

    Spectra here are of shape (stain, channel). Components are named ``'{stain}_PC{i}'`` (i from 0). After
    `collect_PCA_data` the embeddings are in units of their standard deviation across the measurements.

    Attributes
    ----------
    name : str
        Panel name.
    stains : numpy.ndarray of str
        Stain names (the keys of the `SpilloverCollection`), in panel order.
    markers : numpy.ndarray of str
        Marker names, in panel order.
    stain_marker_name : numpy.ndarray of str
        Labels ``'stain | marker'``, in panel order.
    components : numpy.ndarray of numpy.ndarray of int
        Indices of the components used per stain.
    total_components : int
        Number of used components over all stains.
    has_PCA_data : bool
        True after `collect_PCA_data`.
    channels : list of str
        Fluorescence channels of the spectra (after `collect_PCA_data`).
    pca_components : pandas.DataFrame
        All fitted components of all stains, shape (component, channel), scaled to embeddings of standard deviation 1.
    pca_embedding : pandas.DataFrame
        Embedding of every measured spectrum, shape (measurement, component), standardised (NaN where a stain was not
        measured).
    pca_offset : pandas.DataFrame
        Mean spectrum per stain, shape (stain, channel).
    pca_components_stain : pandas.Series
        Stain of every component.
    pca_components_use : pandas.Series of bool
        Whether the panel uses the component.
    pca_explained_variance : pandas.Series
        Explained variance per component (before the standardisation).
    pca_explained_variance_channelratio : pandas.DataFrame
        Share of the variance of every channel explained by each component, shape (component, channel).
    spectra_full : dict of str to pandas.DataFrame
        Measured spectra per stain used in the PCA, each of shape (measurement, channel).
    """
    def __init__(self):
        self.has_PCA_data = False
    
    def __repr__(self):
        repstr = f"PanelConfiguration '{self.name}'. Contains {len(self.stains)} dyes, uses {self.total_components} PC components."
        markers = pd.Series(self.markers, index=self.stains)
        repstr += "\n    "+markers.to_string().replace("\n","\n    ")
        return repstr
    
    @classmethod
    def from_csv(cls, csv_path, name):
        """Read the panel configuration from a .csv file.

        The file needs the columns:

        - ``stain``: name of the dye as registered in the `SpilloverCollection` (a value of its `register_on` column);
        - ``marker``: name of the corresponding marker;
        - ``components``: which principal components of the stain to use, as a string of single digits: ``""`` none,
          ``"01"`` the first two, ``"13"`` the second and fourth, etc.

        Parameters
        ----------
        csv_path : str
            Path of the .csv file.
        name : str
            Name of the panel.

        Returns
        -------
        PanelConfiguration
            Without PCA data (see `collect_PCA_data`).

        Examples
        --------
        >>> pc = PanelConfiguration.from_csv("panel_HD7_TCell.csv", "HD7_TCell")
        >>> pc.collect_PCA_data(spcollection)
        >>> pc.save("panel_HD7_TCell.pt")
        """
        pc = cls()
        pc.name = name
        
        pc.df = pd.read_table(csv_path, sep=",", dtype=str).fillna("")
        pc.stains = pc.df["stain"].to_numpy()
        pc.markers = pc.df["marker"].to_numpy()
        pc.stain_marker_name = (pc.df["stain"] + " | " + pc.df["marker"]).to_numpy()
        pc.components = pc.df["components"].apply(lambda x: np.array(list(x), dtype=int)).to_numpy()
        pc.total_components = sum([len(c) for c in pc.components])
        
        return pc
    
    def export(self):
        """Export all data as a dict for ``torch.save`` (pandas objects expanded with `cytovanni.torch.pdsave_expand_dict`).

        Returns
        -------
        dict
            Input for `from_saved` (``exported``).
        """
        savekeys = [ 'name',
                     'df',
                     'stains',
                     'markers',
                     'stain_marker_name',
                     'components',
                     'total_components',
                     'has_PCA_data',
                     'pca_components',
                     'pca_components_stain',
                     'pca_components_use',
                     'pca_embedding',
                     'pca_explained_variance',
                     'pca_explained_variance_channelratio',
                     'pca_offset',
                     'channels',
                     'spectra_full',
                   ]
        exported = {}
        for key in savekeys:
            if hasattr(self, key):
                if key=="spectra_full":
                    exported[key] = pdsave_expand_dict(getattr(self, key))
                else:
                    exported[key] = getattr(self, key)
        return pdsave_expand_dict(exported)
    
    def save(self, savepath):
        """Save the panel configuration with ``torch.save`` to `savepath` (see `export`)."""
        exported = self.export()
        torch.save(exported, savepath)
    
    @classmethod
    def from_saved(cls, savepath="", exported=None):
        """Load a panel configuration saved with `save`, or from the output of `export`.

        The file is read with ``torch.load(weights_only=False)``, i.e. unpickled: only load files you trust.

        Parameters
        ----------
        savepath : str, default ""
            Path of the saved file.
        exported : dict, optional
            Output of `export`; used if `savepath` is empty.

        Returns
        -------
        PanelConfiguration
        """
        exported = pdsave_collect_dict(torch.load(savepath, weights_only=False) if savepath else exported)
        pc = cls()
        for key in exported:
            if key=="spectra_full":
                setattr(pc, key, pdsave_collect_dict(exported[key]))
            else:
                setattr(pc, key, exported[key])
        
        return pc
    
    def collect_PCA_data(self, spcollection):
        """Collect the PCA fits of the panel's stains from a spillover collection.

        Takes the fit of every stain of the panel from ``spcollection.single_stains`` (after
        `SpilloverCollection.add_PCA_fit`) and sets the ``pca_*``, ``spectra_full`` and ``channels`` attributes. The
        components are rescaled from L2 norm 1 to embeddings with standard deviation 1 (over the measured spectra,
        ddof 0) for every component.

        Parameters
        ----------
        spcollection : SpilloverCollection
            Collection with PCA fits; must contain every stain of the panel.

        Raises
        ------
        ValueError
            If the panel uses a component that was not fitted (a stain with n spectra has at most n-1 components, see
            `SingleSpectrumPCAFitter`, and at most 5).
        """
        exported = [spcollection.single_stains[key].export_pca(self) for key in self.stains]

        self.pca_embedding = pd.concat([e["embedding"] for e in exported], axis=1)
        self.pca_components = pd.concat([e["components"] for e in exported], axis=0)
        self.pca_offset = pd.concat([e["offset"] for e in exported], axis=0)
        self.pca_components_stain = pd.concat([e["components_stain"] for e in exported], axis=0)
        self.pca_components_use = pd.Series(False, index=self.pca_components_stain.index)
        use = [i for e in exported for i in e["components_use"]]
        missing = [i for i in use if i not in self.pca_components_use.index]
        if missing:
            raise ValueError(f"Panel '{self.name}' uses PCA components {missing} that were not fitted: a stain with n spectra "
                             f"has at most n-1 components (and at most N of add_PCA_fit)!")
        self.pca_components_use.loc[use] = True
        self.pca_explained_variance = pd.concat([e["explained_variance"] for e in exported], axis=0)
        self.pca_explained_variance_channelratio = pd.concat([e["explained_variance_channelratio"] for e in exported], axis=0)
        self.spectra_full = {k:e["spectra"] for k, e in zip(self.stains, exported)}
        
        self.channels = self.pca_components.columns.to_list()
        
        # norm embeddings to std 1 for every component
        std = np.nanstd(self.pca_embedding, axis=0)
        self.pca_embedding = self.pca_embedding / std[None]
        self.pca_components = self.pca_components * std[:,None]

        self.has_PCA_data = True
    
    def plot_embedding(self):
        """Plot the embedding of every measured spectrum against the explained variance of its component."""
        var = self.pca_explained_variance.to_numpy()[None].repeat(self.pca_embedding.shape[0], axis=0).flatten()
        emb = self.pca_embedding.to_numpy().flatten()
        mask = self.pca_components_use.to_numpy()[None].repeat(self.pca_embedding.shape[0], axis=0).flatten()

        plt.scatter(emb, var, s=1, label="all PCs")
        plt.scatter(emb[mask], var[mask], s=1, label="used PCs")
        plt.yscale("log")
        plt.xlabel(r"Embedding [$\sigma$]", size=15)
        plt.ylabel("Explained Variance", size=15)
        plt.legend()
    
    def get_run_spectra(self, run, pca_impute=False, label_marker=False):
        """Get the measured spectra of one single-stain measurement for all stains of the panel.

        Parameters
        ----------
        run : str
            Measurement name (index of the spillover collection).
        pca_impute : bool, default False
            Fill stains not measured in `run` with their mean spectrum (``pca_offset``); otherwise NaN.
        label_marker : bool, default False
            Label the rows with ``stain_marker_name`` instead of the stain names.

        Returns
        -------
        pandas.DataFrame
            Spectra, shape (stain, channel), in panel order.
        """
        spectra = {}
        for key in self.spectra_full:
            if run in self.spectra_full[key].index:
                spectra[key] = self.spectra_full[key].loc[run]
            else:
                if pca_impute:
                    spectra[key] = self.pca_offset.loc[key]
                else:
                    spectra[key] = pd.Series(np.nan, index=self.spectra_full[key].columns)
        spectra = pd.DataFrame(spectra).T
        if label_marker:
            spectra = spectra.loc[self.stains]
            spectra.index = self.stain_marker_name
        return spectra
    
    def get_single_spectrum_embedding(self, dye, spectrum):
        """Get the embedding of one spectrum of `dye` in the standardised components of that dye.

        Least-squares projection (pseudo-inverse of the components) of the spectrum minus the mean spectrum of the dye.

        Parameters
        ----------
        dye : str
            Stain of the panel.
        spectrum : pandas.Series
            Spectrum, index the channels.

        Returns
        -------
        pandas.Series
            Embedding per component of `dye` (all fitted components, not only the used ones); empty if `dye` has none.
        """
        spectrum = spectrum.loc[self.channels]
        compinds = self.pca_components_stain.index[self.pca_components_stain==dye]
        if len(compinds)==0:
            embedding = pd.Series()
        else:
            embedding = pd.Series((spectrum.to_numpy()-self.pca_offset.loc[dye]) @ np.linalg.pinv(self.pca_components.loc[compinds]), index=compinds)
        return embedding
    
    def get_spectra_embedding(self, spectra, ensure_complete=False):
        """Get the panel embedding of a set of spectra.

        Parameters
        ----------
        spectra : pandas.DataFrame
            Spectra, shape (stain, channel): index the stain names, columns the channels (at least ``self.channels``).
        ensure_complete : bool, default False
            Raise if the embedding does not cover every fitted component or contains NaN.

        Returns
        -------
        pandas.Series
            Embedding of all fitted components of the given stains (used or not), in the order of ``pca_components``, in
            units of the standard deviation of each component.

        Raises
        ------
        ValueError
            If the index of `spectra` has duplicates or stains that are not in the panel, or, with `ensure_complete`,
            the embedding is incomplete.
        """
        if not spectra.index.is_unique:
            raise ValueError("Spectra don't have a unique index, i.e. there are repeated dyes!")
        missing = [dye not in self.stains for dye in spectra.index]
        if sum(missing)>0:
            raise ValueError(f"Dyes {spectra.index[missing]} are not part of the panel!")
        embedding = []
        for dye in spectra.index:
            embedding.append(self.get_single_spectrum_embedding(dye, spectra.loc[dye]))
        embedding = [e for e in embedding if len(e)>0]
        embedding = pd.concat(embedding) if len(embedding)>0 else pd.Series(dtype=float) # all-fixed panel: no components
        embedding = embedding.loc[[i for i in self.pca_components.index if i in embedding.index]]

        if ensure_complete:
            if (len(set(self.pca_components.index)-set(embedding.index))>0) or (embedding.isna().sum()>0):
                raise ValueError("The resulting embedding is not complete, i.e. either spectra are missing or some spectra contain NaN!")
        return embedding

    def get_spectra_from_embedding(self, embedding, only_included_components=True, allow_missing=False, clip_negative=True):
        """Turn a panel embedding into the spectra of all stains of the panel.

        Parameters
        ----------
        embedding : pandas.Series
            Embedding per component, as returned by `get_spectra_embedding`.
        only_included_components : bool, default True
            Only use the components the panel uses (the others contribute 0); otherwise all fitted components.
        allow_missing : bool, default False
            Return NaN spectra for stains whose components are missing from `embedding`; otherwise raise.
        clip_negative : bool, default True
            Set negative spectrum entries to 0.

        Returns
        -------
        pandas.DataFrame
            Spectra, shape (stain, channel), index ``self.stains``. Stains without components get their mean spectrum.

        Raises
        ------
        ValueError
            If the embedding of a stain is missing and not `allow_missing`.
        """
        spectra = []
        for dye in self.stains:
            compind = self.pca_components_stain.index[self.pca_components_stain==dye]
            components = self.pca_components.loc[compind]
            components_mask = self.pca_components_use.loc[compind].astype(float) if only_included_components else 1.
            try:
                emb = embedding.loc[compind]
                if only_included_components: # unused components contribute 0, also if their embedding is not finite
                    emb = emb.where(components_mask!=0, 0.)
                spectrum = self.pca_offset.loc[dye] + (components_mask * emb) @ components
            except KeyError:
                if allow_missing:
                    spectrum = pd.Series(np.nan, index=components.columns)
                else:
                    raise ValueError(f"Missing embedding for dye {dye}!")
            spectra.append(spectrum)
        spectra = pd.DataFrame(spectra, index=self.stains)
        if clip_negative:
            spectra = np.clip(spectra, a_min=0, a_max=None)
        return spectra

    def plot_test_spectra_reconstruction(self, spectra, linthresh=1e-2):
        """Plot the reconstruction error of spectra from their embedding, with all fitted and with the used components.

        Gray lines mark errors of 100% and 10% of the original entry.

        Parameters
        ----------
        spectra : pandas.DataFrame
            Spectra, shape (stain, channel), e.g. from `get_run_spectra`.
        linthresh : float, default 1e-2
            Linear range of the symmetric log axes.

        Returns
        -------
        pandas.Series
            Absolute reconstruction error with the used components, index ``'stain | channel'``.
        """
        spectra = spectra.loc[:, self.channels]
        embedding = self.get_spectra_embedding(spectra)
        # reconstructions are indexed by self.stains: align them to the rows/columns of spectra
        spectra_smoothed = self.get_spectra_from_embedding(embedding, only_included_components=True, allow_missing=True).loc[spectra.index, spectra.columns]
        spectra_smoothed_full = self.get_spectra_from_embedding(embedding, only_included_components=False, allow_missing=True).loc[spectra.index, spectra.columns]
    
        fig, ax = plt.subplots(1, 2, figsize=(15,6))
    
        ax[0].scatter(spectra.to_numpy().flatten(), (spectra_smoothed_full.to_numpy().flatten()-spectra.to_numpy().flatten()), s=.1)
        ax[0].set_xscale("symlog", linthresh=linthresh)
        ax[0].set_ylim([min(-1, ax[0].get_ylim()[0]), max(1, ax[0].get_ylim()[1])])
        ax[0].set_yscale("symlog", linthresh=linthresh)
        x = np.logspace(-4, np.log10(ax[0].get_ylim()[1]), 1000)
        ax[0].plot(x, x, color="gray", linewidth=.5)
        ax[0].plot(x, .1*x, color="gray", linewidth=.5)
        x = np.logspace(-4, np.log10(-ax[0].get_ylim()[0]), 1000)
        ax[0].plot(x, -x, color="gray", linewidth=.5)
        ax[0].plot(x, -.1*x, color="gray", linewidth=.5)
        ax[0].set_title("All Principal Components", size=25)
        ax[0].set_xlabel("Original Intensity", size=15)
        ax[0].set_ylabel("Reconstruction Error", size=15)
    
        ax[1].scatter(spectra.to_numpy().flatten(), (spectra_smoothed.to_numpy().flatten()-spectra.to_numpy().flatten()), s=.1)
        ax[1].set_xscale("symlog", linthresh=linthresh)
        ax[1].set_ylim([min(-1, ax[1].get_ylim()[0]), max(1, ax[1].get_ylim()[1])])
        ax[1].set_yscale("symlog", linthresh=linthresh)
        x = np.logspace(-4, np.log10(ax[1].get_ylim()[1]), 1000)
        ax[1].plot(x, x, color="gray", linewidth=.5)
        ax[1].plot(x, .1*x, color="gray", linewidth=.5)
        x = np.logspace(-4, np.log10(-ax[1].get_ylim()[0]), 1000)
        ax[1].plot(x, -x, color="gray", linewidth=.5)
        ax[1].plot(x, -.1*x, color="gray", linewidth=.5)
        ax[1].set_title("Chosen Principal Components", size=25)
        ax[1].set_xlabel("Original Intensity", size=15)
        ax[1].set_ylabel("Reconstruction Error", size=15)
    
        diff = spectra_smoothed.to_numpy().flatten()-spectra.to_numpy().flatten()
        diff = pd.Series(np.abs(diff), index=[i + " | " + j for i in spectra.index for j in spectra.columns])
        return diff


