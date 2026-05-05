import torch

from ..exceptions import SpilloverInversionException


def torch_invert_spectra(spectra, use_pinv=False):
    """ Invert spectra of shape (..., stain, channel)
        returns (..., channel, stain)
        
        Can use torch.linalg.pinv, but this might have issues with the gradient for repeated eigenvalues?
        https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
        Probably better to invert by hand, always has to work for correct panels.
    """
    if use_pinv:
        return torch.linalg.pinv(spectra)
    else:
        if spectra.shape[-1]==spectra.shape[-2]: # normal flow, square spillover
            return torch.linalg.inv(spectra)
        elif spectra.shape[-1]>spectra.shape[-2]: # spectral flow, more channels than stains
            # get inverse of S as S.T (S S.T)^(-1)
            return spectra.transpose(-1,-2) @ torch.linalg.inv(spectra @ spectra.transpose(-1,-2))
        else: # not invertible
            raise SpilloverInversionException(
                f"Spillover matrix with {spectra.shape[-2]} stains and {spectra.shape[-1]} channels is not invertible! Needs more channels than stains.")

def torch_apply_unmixing_inv(inv_spectra, x):
        """ Apply unmixing to x (..., channel), using the inverted spectra (..., channel, stain) directly.
            returns (..., stain)
        """
        return torch.einsum("...ij, ...i -> ...j", inv_spectra, x)

def torch_apply_unmixing_lstsq(spectra, x):
    """ Apply unmixing to x (..., channel), using the spectra (..., channel, stain) and solving a least squares problem.
        More stable than explicit inversion, though probably much slower if most x share the same matrix.
        returns (..., stain)
        
        The additional indices (...) need to be of the same dimension for the spectra and x, and broadcastable.
        The inv unmixing tolerates batch dimensions in x that are missing in spectra etc., this is not the case here.
        
        Be careful with this! Will throw unexplained memory errors if trying to unmix too many events at once.
    """
    return torch.linalg.lstsq(spectra.transpose(-1,-2), x[...,None]).solution[...,0]
