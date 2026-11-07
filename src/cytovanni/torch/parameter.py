"""Parameters that can be extended by new entries (e.g. new batches) while the existing entries stay frozen."""
import numpy as np
import torch
from torch.nn import Module, Parameter, ModuleDict
import torch.nn.functional as F

class ExtendableParameter(Module):
    """Parameter tensor that can be extended along one axis, e.g. by new batches, freezing the existing entries.

    Used for per-batch parameters (log scale factors, panel embeddings, rainbow model parameters): when new batches
    are added to a trained model, `extend` adds entries for them and, by default, only those are trained afterwards.

    The value (`forward`) is ``param_fixed + param_mask_trainable * param_trainable``; with `axis_norm`, the mean
    along that axis is subtracted. Only `param_trainable` receives gradients; the other parts are non-trainable
    ``Parameter`` objects, so all of them are in the state dict. The constructor arguments (`init_value`, `axis_norm`,
    ...) are not in the state dict: recreate the object with the same arguments (and the same `extend` calls) before
    loading a state dict.

    Parameters
    ----------
    shape : tuple of int
        Initial shape.
    axis_extend : int
        Axis along which `extend` adds entries (usually the batch axis).
    init_value : float or torch.Tensor
        Initial value of the trainable part; a tensor must broadcast to `shape` (and to the shape of the entries
        added by `extend`).
    allow_grad : bool, default True
        Whether the trainable part can receive gradients at all (`requires_grad_` cannot enable it otherwise).
    axis_norm : int, optional
        If given, the value is normalised to mean 0 along this axis (e.g. log factors with geometric mean 1). The
        mean runs over the entries marked in `param_mask_meannorm` (all at construction; entries added by `extend`
        without `keep_prior_trainable_structure` are excluded) and is subtracted from all entries. Entries without
        training data (e.g. a (batch, channel) combination never measured) are not excluded automatically: they keep
        their initial value and still enter the mean, so the common zero point depends on them. Differences between
        entries with data are not affected.

    Examples
    --------
    >>> p = ExtendableParameter((3,), 0, 0., axis_norm=0)   # log factors of 3 batches
    >>> p.extend([3, 4])                                     # two new batches; the first three are frozen
    >>> p().shape
    torch.Size([5])
    """
    def __init__(self, shape, axis_extend, init_value, allow_grad=True, axis_norm=None):
        super().__init__()
        
        self.init_value = init_value
        self.axis_extend = axis_extend
        self.allow_grad = allow_grad
        self.axis_norm = axis_norm
        
        # fixed part
        self.param_fixed = Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)
        # mask to only train relevant trainable part
        self.param_mask_trainable = Parameter(self.get_initial_trainable_mask(shape), requires_grad=False)
        # trainable part
        self.param_trainable = Parameter(self.get_initial_trainable(shape), requires_grad=allow_grad)
        # mask which part should be used for mean calculation
        if axis_norm is not None:
            self.param_mask_meannorm = Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False)
    
    def __repr__(self):
        """Show the shape and the fixed, trainable and mask parts."""
        retstr = f"Extendable Parameter of shape {tuple(self.shape)} containing:"
        retstr += "\n    Fixed Part, "
        retstr += self.param_fixed.__repr__().replace("\n","\n    ")
        retstr += "\n    Trainable Part, "
        retstr += self.param_trainable.__repr__().replace("\n","\n    ")
        retstr += "\n    Trainable Mask, "
        retstr += self.param_mask_trainable.__repr__().replace("\n","\n    ")
        if self.axis_norm is not None:
            retstr += "\n    Mean Norm Mask, "
            retstr += self.param_mask_meannorm.__repr__().replace("\n","\n    ")
        return retstr
    
    def get_initial_trainable(self, shape, device="cpu"):
        """Return the initial trainable values for `shape` (`init_value` broadcast); override in subclasses.

        Parameters
        ----------
        shape : tuple of int
            Shape of the part to initialise (the full shape, or the added part in `extend`).
        device : str or torch.device, default "cpu"
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor of shape `shape`, float32

        Raises
        ------
        RuntimeError
            If `init_value` does not broadcast to `shape`.
        """
        try:
            return torch.zeros(shape, dtype=torch.float32, device=device) + torch.as_tensor(self.init_value, dtype=torch.float32, device=device)
        except Exception as e:
            raise RuntimeError("Make sure the initial value can be properly broadcast to the full parameter shape!") from e
    
    def get_initial_trainable_mask(self, shape, device="cpu"):
        """Return the initial trainable mask for `shape` (all ones: every entry trainable); override in subclasses."""
        return torch.ones(shape, dtype=torch.float32, device=device)
    
    def set_trainable_part(self, val, sl=slice(None)):
        """Overwrite the trainable values, e.g. with an initialisation, leaving frozen entries unchanged.

        Only entries whose trainable mask is non-zero are overwritten, so this can be called after `extend` or
        `freeze_selective` to initialise just the new or still trainable entries. Also stores `val` and `sl` as
        attributes ``self.val``, ``self.sl``.

        Parameters
        ----------
        val : torch.Tensor
            New values; must broadcast to ``param_trainable[sl]``. These are the trainable part, before the fixed
            part is added and before the mean normalisation of `forward`.
        sl : index expression, default slice(None)
            Part of the parameter to set, e.g. ``(slice(None), [3, 4])``.
        """
        self.val = val
        self.sl = sl
        mask = self.param_mask_trainable.data!=0.
        mask_full = torch.full_like(mask, False)
        mask_full[sl] = mask[sl]
        data_new = self.param_trainable.data.clone()
        data_new[sl] = val.to(torch.float32).to(self.param_trainable.data.device)
        self.param_trainable.data[mask_full] = data_new[mask_full]
    
    @property
    def shape(self):
        """Current shape of the parameter (torch.Size)."""
        return self.param_fixed.shape
    
    @property
    def device(self):
        """Device the parameter is on."""
        return self.param_fixed.device
    
    def forward_param_sum(self):
        """Return ``param_fixed + param_mask_trainable * param_trainable`` (the value before the mean normalisation)."""
        return self.param_fixed + self.param_mask_trainable * self.param_trainable
    
    def forward(self):
        """Return the current value, shape `shape`: `forward_param_sum`, minus the masked mean along `axis_norm`."""
        value = self.forward_param_sum()
        if self.axis_norm is not None:
            mean = (value * self.param_mask_meannorm).sum(axis=self.axis_norm, keepdims=True)
            mean = mean / self.param_mask_meannorm.sum(axis=self.axis_norm, keepdims=True)
            value = value - mean
        return value
    
    def requires_grad_(self, requires_grad=True):
        """Enable or disable gradients for the trainable part (only if `allow_grad`); the other parts never get any.

        Returns
        -------
        ExtendableParameter
            self.
        """
        self.param_fixed.requires_grad_(False)
        self.param_mask_trainable.requires_grad_(False)
        self.param_trainable.requires_grad_(requires_grad and self.allow_grad)
        if self.axis_norm is not None:
            self.param_mask_meannorm.requires_grad_(False)
        return self
    
    def extend(self, inds, keep_prior_trainable_structure=False):
        """Extend the parameter along `axis_extend` so it contains the indices `inds`; new entries get `init_value`.

        The new length is ``max(inds) + 1`` (unchanged if that is not larger); `inds` are positions, e.g. the new
        batch indices returned by ``MultiOrdinalEncoder.extend``.

        Parameters
        ----------
        inds : list of int
            Indices that must exist afterwards (empty: no new entries, see `freeze`).
        keep_prior_trainable_structure : bool, default False
            If False, the current values (`forward_param_sum`) of all existing entries are moved into the fixed part
            and frozen, only the new entries are trainable, and the mean normalisation keeps running over the
            previously used entries only (the new entries do not enter the mean), so the zero point stays as fitted.
            If True, the existing entries stay as they are (trainable where they were), and the new entries are
            trainable and enter the mean normalisation.
        """
        addshape = list(self.param_fixed.shape)
        addshape[self.axis_extend] = max((max(inds)+1 if len(inds)>0 else 0) - addshape[self.axis_extend], 0)
        
        if keep_prior_trainable_structure:
            # Add zeros to fixed part
            self.param_fixed = Parameter(torch.cat([self.param_fixed.data,
                                                    torch.zeros(addshape, dtype=torch.float32, device=self.param_fixed.device)],
                                                   axis=self.axis_extend),
                                         requires_grad=False)
            # add ones to trainable mask
            self.param_mask_trainable = Parameter(torch.cat([self.param_mask_trainable.data,
                                                             self.get_initial_trainable_mask(addshape, device=self.param_mask_trainable.device)],
                                                   axis=self.axis_extend),
                                         requires_grad=False)
            # Add init_value to trainable part, keep requires_grad
            self.param_trainable = Parameter(torch.cat([self.param_trainable.data,
                                                        self.get_initial_trainable(addshape, device=self.param_trainable.device)],
                                                   axis=self.axis_extend),
                                         requires_grad=self.param_trainable.requires_grad)
            if self.axis_norm is not None:
                # add ones to mean norm mark
                self.param_mask_meannorm = Parameter(torch.cat([self.param_mask_meannorm.data,
                                                                 torch.ones(addshape, dtype=torch.float32, device=self.param_mask_meannorm.device)],
                                                       axis=self.axis_extend),
                                             requires_grad=False)
        else:
            # New fixed part is full old plus zeros at new
            self.param_fixed = Parameter(torch.cat([self.forward_param_sum().detach(),
                                                    torch.zeros(addshape, dtype=torch.float32, device=self.param_fixed.device)],
                                                   axis=self.axis_extend),
                                         requires_grad=False)
            # New mask is zeros for all old, ones for new
            self.param_mask_trainable = Parameter(torch.cat([torch.zeros_like(self.param_mask_trainable.data),
                                                             self.get_initial_trainable_mask(addshape, device=self.param_mask_trainable.device)],
                                                   axis=self.axis_extend),
                                         requires_grad=False)
            # New trainable is zeros for old, init_value for new
            self.param_trainable = Parameter(torch.cat([torch.zeros_like(self.param_trainable.data),
                                                        self.get_initial_trainable(addshape, device=self.param_trainable.device)],
                                                   axis=self.axis_extend),
                                         requires_grad=self.param_trainable.requires_grad)
            if self.axis_norm is not None:
                # add zeros to mean norm mark, only use the norm the way it was trained initially
                self.param_mask_meannorm = Parameter(torch.cat([self.param_mask_meannorm.data,
                                                                 torch.zeros(addshape, dtype=torch.float32, device=self.param_mask_meannorm.device)],
                                                       axis=self.axis_extend),
                                             requires_grad=False)
    
    def freeze(self):
        """Freeze all entries: move the current values into the fixed part (``extend([])``)."""
        self.extend([])
    
    def freeze_selective(self, sl):
        """Freeze the entries indexed by `sl`: their current value moves into the fixed part and is no longer trained.

        Parameters
        ----------
        sl : index expression
            Entries to freeze, e.g. ``(slice(None), [0, 1])`` for batches 0 and 1 along axis 1.
        """
        current_sum = self.forward_param_sum().detach()
        self.param_mask_trainable.data[sl] = 0.
        self.param_trainable.data[sl] = 0.
        self.param_fixed.data[sl] = current_sum[sl]

    def add_to_fixed(self, values, sl=None):
        """Add values to the fixed part of the parameter, in place.

        Parameters
        ----------
        values : torch.Tensor
            Values to add; must broadcast to ``param_fixed[sl]`` (or to ``param_fixed`` without `sl`).
        sl : index expression, optional
            Part to add to. Default: the whole parameter.
        """
        if sl is None:
            self.param_fixed.data += values.to(dtype=torch.float32, device=self.param_fixed.device)
        else:
            self.param_fixed.data[sl] += values.to(dtype=torch.float32, device=self.param_fixed.device)


class PrecisionExtendableParameter(ExtendableParameter):
    r"""ExtendableParameter holding a factor A of a precision matrix, :math:`P = A A^T` (rainbow bead model).

    The last two axes are the matrix axes, shape (..., D, D); A itself is unconstrained. `forward` returns the
    precision and its log-determinant, computed as :math:`2 \log\det A` (since :math:`\det A^T = \det A`), which is
    more stable than the log-determinant of :math:`A A^T`. A must stay invertible: it is initialised as a multiple of
    the identity, and in a likelihood fit the log-determinant term goes to -inf for a singular A, which usually keeps
    it invertible for small enough learning rates. `axis_norm` is not applied.

    Parameters are those of `ExtendableParameter`; `shape` needs at least two axes, and `init_value` (float) is the
    initial diagonal of A, so the initial precision is ``init_value**2`` times the identity.

    Raises
    ------
    ValueError
        On construction, if `shape` has fewer than two axes.
    """
    def forward(self):
        """Return the precision and its log-determinant.

        Returns
        -------
        precision : torch.Tensor of shape (..., D, D)
            ``A @ A^T``.
        logdet : torch.Tensor of shape (...)
            ``log det(A @ A^T)``, computed as ``2 * logdet(A)``; where that is NaN (e.g. negative determinant), the
            log-determinant of the precision is used instead, then ``2 * slogdet(A).logabsdet``.

        Raises
        ------
        ValueError
            If the log-determinant is still NaN (singular precision matrix).
        """
        value = self.forward_param_sum()
        precision = torch.einsum("...ij, ...jk -> ...ik", value, value.transpose(-1,-2))
        logdet = 2 * torch.logdet(value)
        
        # if logdet contains nans, try going through precision directly
        if torch.any(torch.isnan(logdet)).item():
            mask = torch.isnan(logdet)
            logdet[mask] = torch.logdet(precision[mask])
        
            # if logdet still contains nans, maybe weird error where the sign fluctuates based on numerical errors
            # use slogdet and ignore sign
            if torch.any(torch.isnan(logdet)).item():
                mask = torch.isnan(logdet)
                logdet[mask] = 2 * torch.slogdet(value[mask]).logabsdet
                
                # if still bad give up and raise error
                if torch.any(torch.isnan(logdet)).item():
                    raise ValueError("Singular precision matrix!")
        
        return precision, logdet
    
    def get_initial_trainable(self, shape, device="cpu"):
        """Return `init_value` times the identity on the last two axes, shape `shape` (ValueError for < 2 axes)."""
        if len(shape)<2:
            raise ValueError(f"Covariance decomposition shape must be at least two dimensions, not {shape}!")
        elif len(shape)==2:
            return torch.eye(*shape, dtype=torch.float32, device=device) * self.init_value
        else:
            prefactor = torch.ones( list(shape)[:-2] + [1,1], dtype=torch.float32, device=device) * self.init_value
            eye = torch.eye( *list(shape)[-2:], dtype=torch.float32, device=device).view( [1]*(len(shape)-2) + list(shape[-2:]) )
            return (prefactor * eye)

class CovarianceTrilExtendableParameter(ExtendableParameter):
    """ExtendableParameter holding the lower-triangular Cholesky factor L of a covariance matrix (scale_tril).

    An alternative to `PrecisionExtendableParameter` for the rainbow bead model (not used by default). Only the lower
    triangle is trainable; `forward` takes the lower triangle and passes the diagonal through a softplus, so it is
    positive. The raw parameter is initialised to `init_value` times the identity, so the initial diagonal of L is
    ``softplus(init_value)`` (about 0.80 for 0.2), not `init_value`. `axis_norm` is not applied.

    Parameters are those of `ExtendableParameter`; `shape` (..., D, D) needs at least two axes.

    Raises
    ------
    ValueError
        On construction, if `shape` has fewer than two axes.
    """
    
    def forward(self):
        """Return the Cholesky factor L, shape (..., D, D): lower triangle of the value, softplus on the diagonal."""
        value = torch.tril( self.param_fixed + self.param_mask_trainable * self.param_trainable )
        value[..., range(value.shape[-2]), range(value.shape[-1])] = F.softplus(value[..., range(value.shape[-2]), range(value.shape[-1])])
        return value
    
    def get_initial_trainable(self, shape, device="cpu"):
        """Return `init_value` times the identity on the last two axes, on `device`."""
        if len(shape)<2:
            raise ValueError(f"Covariance decomposition shape must be at least two dimensions, not {shape}!")
        elif len(shape)==2:
            return torch.eye(*shape, dtype=torch.float32, device=device) * self.init_value
        else:
            prefactor = torch.ones( list(shape)[:-2] + [1,1], dtype=torch.float32, device=device) * self.init_value
            eye = torch.eye( *list(shape)[-2:], dtype=torch.float32, device=device).view( [1]*(len(shape)-2) + list(shape[-2:]) )
            return (prefactor * eye)
    
    def get_initial_trainable_mask(self, shape, device="cpu"):
        """Return the trainable mask: ones on and below the diagonal, on `device`."""
        return torch.tril(torch.ones(shape, dtype=torch.float32, device=device))

