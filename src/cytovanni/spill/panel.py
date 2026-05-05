import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict

class PanelConfiguration():
    def __init__(self):
        self.has_PCA_data = False
    
    def __repr__(self):
        repstr = f"PanelConfiguration '{self.name}'. Contains {len(self.stains)} dyes, uses {self.total_components} PC components."
        markers = pd.Series(self.markers, index=self.stains)
        repstr += "\n    "+markers.to_string().replace("\n","\n    ")
        return repstr
    
    @classmethod
    def from_csv(cls, csv_path, name):
        """ Reads panel configuration from .csv file.
            Should have the following columns:
                - stain: name of the fluorophore as it should be registered to the name in the SpilloverCollection.
                - marker: name of the corresponding marker
                - components: which PCs should be used for each dye.
                              "" is none, "01" the first two, "13" the second and fourth etc.
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
        """ Export all data of self.
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
        """ Save self to savepath.
        """
        exported = self.export()
        torch.save(exported, savepath)
    
    @classmethod
    def from_saved(cls, savepath="", exported=None):
        """ Load from saved config.
            Either load from savepath or use dict exported.
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
        """ Collect all PCA data from spillover collection spcollection.
            Also changes the norm from L2-norm 1 of the components, to std 1 for the embeddings.
        """
        exported = [spcollection.single_stains[key].export_pca(self) for key in self.stains]

        self.pca_embedding = pd.concat([e["embedding"] for e in exported], axis=1)
        self.pca_components = pd.concat([e["components"] for e in exported], axis=0)
        self.pca_offset = pd.concat([e["offset"] for e in exported], axis=0)
        self.pca_components_stain = pd.concat([e["components_stain"] for e in exported], axis=0)
        self.pca_components_use = pd.Series(False, index=self.pca_components_stain.index)
        self.pca_components_use.loc[[i for e in exported for i in e["components_use"]]] = True
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
        """ Plot embedding against explained variance.
        """
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
        """ Get spectra measured in one run.
            
            Can impute missing spectra by the offset of the PCA fit, i.e. the mean across all runs.

            :param label_marker: bool. If True, label output with self.stain_marker_name, else with stain names.
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
        """ Get embedding for spectrum of single dye.
        """
        spectrum = spectrum.loc[self.channels]
        compinds = self.pca_components_stain.index[self.pca_components_stain==dye]
        if len(compinds)==0:
            embedding = pd.Series()
        else:
            embedding = pd.Series((spectrum.to_numpy()-self.pca_offset.loc[dye]) @ np.linalg.pinv(self.pca_components.loc[compinds]), index=compinds)
        return embedding
    
    def get_spectra_embedding(self, spectra, ensure_complete=False):
        """ Get PCA embedding of spectra.
    
            :param spectra: pd.DataFrame. Spectra, should have the channels as columns, and the dye name as index.
            
            :param ensure_complete: bool. If this is set, raises an error if the resulting embedding is not complete and free of NaNs.
        """
        if not spectra.index.is_unique:
            raise ValueError("Spectra don't have a unique index, i.e. there are repeated dyes!")
        missing = [dye not in self.stains for dye in spectra.index]
        if sum(missing)>0:
            raise ValueError(f"Dyes {spectra.index[missing]} are not part of the panel!")
        embedding = []
        for dye in spectra.index:
            embedding.append(self.get_single_spectrum_embedding(dye, spectra.loc[dye]))
        embedding = pd.concat([e for e in embedding if len(e)>0])
        embedding = embedding.loc[[i for i in self.pca_components.index if i in embedding.index]]

        if ensure_complete:
            if (len(set(self.pca_components.index)-set(embedding.index))>0) or (embedding.isna().sum()>0):
                raise ValueError("The resulting embedding is not complete, i.e. either spectra are missing or some spectra contain NaN!")
        return embedding

    def get_spectra_from_embedding(self, embedding, only_included_components=True, allow_missing=False, clip_negative=True):
        """ Turn PCA embedding into full panel spectra.
    
            :param embedding: pd.Series. Series of PCA embeddings as produced by self.get_spectra_embedding.
    
            :param only_included_components: bool. Whether to only use the components that were included in the panel, or all that were fitted.
    
            :param allow_missing: bool. Will throw an error if some embeddings are missing, this can be ignored and instead output nan for the corresponding spectra.
    
            :param clip_negative: bool. Whether to clip negative entries to zero. For realistic embeddings this isn't a big issues, but still nicer to have strictly positive spectra.
        """
        spectra = []
        for dye in self.stains:
            compind = self.pca_components_stain.index[self.pca_components_stain==dye]
            components = self.pca_components.loc[compind]
            components_mask = self.pca_components_use.loc[compind].astype(float) if only_included_components else 1.
            try:
                spectrum = self.pca_offset.loc[dye] + (components_mask * embedding.loc[compind]) @ components
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
        """ Plot spectra against their reconstruction error, using either the full PCA or only the chosen components.
            Also returns a list of all absolute reconstruction errors.
        """
        spectra_smoothed = self.get_spectra_from_embedding(self.get_spectra_embedding(spectra), only_included_components=True)
        spectra_smoothed_full = self.get_spectra_from_embedding(self.get_spectra_embedding(spectra), only_included_components=False)
    
        fig, ax = plt.subplots(1, 2, figsize=(15,6))
    
        ax[0].scatter(spectra.to_numpy().flatten(), (spectra_smoothed_full.to_numpy().flatten()-spectra.to_numpy().flatten()), s=.1)
        ax[0].set_xscale("symlog", linthresh=linthresh)
        ax[0].set_ylim([min(-1, ax[0].get_ylim()[0]), max(1, ax[0].get_ylim()[1])])
        ax[0].set_yscale("symlog", linthresh=linthresh)
        ax[0].set_title("All Principal Components", size=25)
        ax[0].set_xlabel("Original Intensity", size=15)
        ax[0].set_ylabel("Reconstruction Error", size=15)
    
        ax[1].scatter(spectra.to_numpy().flatten(), (spectra_smoothed.to_numpy().flatten()-spectra.to_numpy().flatten()), s=.1)
        ax[1].set_xscale("symlog", linthresh=linthresh)
        ax[1].set_ylim([min(-1, ax[1].get_ylim()[0]), max(1, ax[1].get_ylim()[1])])
        ax[1].set_yscale("symlog", linthresh=linthresh)
        ax[1].set_title("Chosen Principal Components", size=25)
        ax[1].set_xlabel("Original Intensity", size=15)
        ax[1].set_ylabel("Reconstruction Error", size=15)
    
        diff = spectra_smoothed.to_numpy().flatten()-spectra.to_numpy().flatten()
        diff = pd.Series(np.abs(diff), index=[i + " | " + j for i in spectra.index for j in spectra.columns])
        return diff


