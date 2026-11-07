"""Differentiable unmixing in torch: invert spectra and apply them to raw intensities.

Spectra are (..., stain, channel) here, as in the panel modules; inverted spectra are (..., channel, stain). Unmixing
is ordinary least squares: for spectral flow (more channels than stains) the inverse is the Moore-Penrose right
inverse ``S.T (S S.T)^-1``.
"""
import torch

from ..exceptions import SpilloverInversionException


def torch_invert_spectra(spectra, use_pinv=False):
    """Invert spectra for unmixing.

    Parameters
    ----------
    spectra : torch.Tensor of shape (..., stain, channel)
        Spectra, one row per stain; leading axes are batch axes.
    use_pinv : bool, default False
        Use ``torch.linalg.pinv`` instead of the explicit formulas below. Its gradient can be problematic for
        (nearly) repeated singular values (see the torch documentation of ``torch.linalg.pinv``), so the explicit
        inverse is the default.

    Returns
    -------
    torch.Tensor of shape (..., channel, stain)
        Inverted spectra, to be used with `torch_apply_unmixing_inv`. For as many channels as stains this is the
        matrix inverse ``inv(S)``; for more channels than stains the right inverse ``S.T @ inv(S @ S.T)``, i.e. the
        unweighted least-squares unmixing.

    Raises
    ------
    SpilloverInversionException
        If there are fewer channels than stains (and not `use_pinv`). A singular matrix raises torch's own error.
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
        """Unmix intensities with inverted spectra.

        Parameters
        ----------
        inv_spectra : torch.Tensor of shape (..., channel, stain)
            Inverted spectra, e.g. from `torch_invert_spectra`.
        x : torch.Tensor of shape (..., channel)
            Intensities on a linear scale (raw or calibrated). The leading axes of `x` and `inv_spectra` broadcast,
            so one matrix can be applied to many events, e.g. `x` of shape (events, channel) with `inv_spectra` of
            shape (channel, stain), or one matrix per event with shape (events, channel, stain).

        Returns
        -------
        torch.Tensor of shape (..., stain)
            Unmixed abundances on the same linear scale, ``x @ inv_spectra``.
        """
        return torch.einsum("...ij, ...i -> ...j", inv_spectra, x)

def torch_apply_unmixing_lstsq(spectra, x):
    """Unmix intensities by solving a least-squares problem per event.

    Gives the same result as `torch_invert_spectra` + `torch_apply_unmixing_inv`, but solves with
    ``torch.linalg.lstsq`` instead of forming the inverse: numerically more stable, but probably much slower when
    most events share the same matrix.

    Parameters
    ----------
    spectra : torch.Tensor of shape (..., stain, channel)
        Spectra, one row per stain.
    x : torch.Tensor of shape (..., channel)
        Intensities on a linear scale (raw or calibrated). Unlike `torch_apply_unmixing_inv`, `spectra` needs the same
        leading axes as `x` (size 1 is enough where they broadcast): for `x` of shape (events, channel) pass spectra of
        shape (events, stain, channel) or (1, stain, channel); a single (stain, channel) matrix raises a RuntimeError.

    Returns
    -------
    torch.Tensor of shape (..., stain)
        Unmixed abundances on the same linear scale.

    Notes
    -----
    Memory use grows quickly with the number of events; unmixing too many events at once can fail with memory
    errors that do not point to the cause, so unmix in chunks.
    """
    return torch.linalg.lstsq(spectra.transpose(-1,-2), x[...,None]).solution[...,0]
