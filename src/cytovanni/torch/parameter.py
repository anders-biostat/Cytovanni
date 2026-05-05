import numpy as np
import torch
from torch.nn import Module, Parameter, ModuleDict
import torch.nn.functional as F

class ExtendableParameter(Module):
    """ Parameter that can be extended along an axis, freezing the prior entries.
        Useful if new batches etc. are added to a model, and only their parameters should get trained afterwards.
        
        Maybe also add initial value etc. as parameter so they get saved and loaded with the state dict?
    """
    def __init__(self, shape, axis_extend, init_value, allow_grad=True, axis_norm=None):
        """ 
            
            :param shape: tuple. Shape of the parameter
            
            :param axis_extend: int. Axis along which to extend.
            
            :param init_value: float, torch.Tensor. Initial value for the parameter. If tensor, needs to be broadcastable to the final shape.
            
            :param allow_grad: bool. Whether it should be trainable.
            
            :param axis_norm: int, None. Normal parameter if None, else normed to mean zero along axis axis_norm.
        """
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
        """ Function to initialise trainable part, overwrite this to make custom parameters.
        """
        try:
            return torch.zeros(shape, dtype=torch.float32, device=device) + torch.as_tensor(self.init_value, dtype=torch.float32, device=device)
        except Exception as e:
            raise RuntimeError("Make sure the initial value can be properly broadcast to the full parameter shape!") from e
    
    def get_initial_trainable_mask(self, shape, device="cpu"):
        """ Function to initialise trainable mask part, overwrite this to make custom parameters.
        """
        return torch.ones(shape, dtype=torch.float32, device=device)
    
    def set_trainable_part(self, val, sl=slice(None)):
        """ Set trainable part to val, optionally indexed with sl.
            Only overwrites the part that is actually trainable to allow adding initialisations after freezing.
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
        return self.param_fixed.shape
    
    @property
    def device(self):
        return self.param_fixed.device
    
    def forward_param_sum(self):
        return self.param_fixed + self.param_mask_trainable * self.param_trainable
    
    def forward(self):
        value = self.forward_param_sum()
        if self.axis_norm is not None:
            mean = (value * self.param_mask_meannorm).sum(axis=self.axis_norm, keepdims=True)
            mean = mean / self.param_mask_meannorm.sum(axis=self.axis_norm, keepdims=True)
            value = value - mean
        return value
    
    def requires_grad_(self, requires_grad=True):
        self.param_fixed.requires_grad_(False)
        self.param_mask_trainable.requires_grad_(False)
        self.param_trainable.requires_grad_(requires_grad and self.allow_grad)
        if self.axis_norm is not None:
            self.param_mask_meannorm.requires_grad_(False)
    
    def extend(self, inds, keep_prior_trainable_structure=False):
        """ Extends self to contain all indices in inds, initialized as at the beginning.
            If keep_prior_trainable_structure, keeps everything as before and only adds new part, mean norm is applied on all new parts.
            If not, shift everything from before into fixed part and freeze, only make new part trainable, mean norm is also frozen to original part.
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
        """ Wrapper on extend, effectively just freezes all parameters.
        """
        self.extend([])
    
    def freeze_selective(self, sl):
        """ Freeze part of self that is indexed by sl.
        """
        current_sum = self.forward_param_sum().detach()
        self.param_mask_trainable.data[sl] = 0.
        self.param_trainable.data[sl] = 0.
        self.param_fixed.data[sl] = current_sum[sl]


class PrecisionExtendableParameter(ExtendableParameter):
    r"""ExtendableParameter, but initialised as the covariance decomposition for the rainbow GMM model.
        
        Encodes an arbitrary matrix A, AA^T is the precision.
        A needs to be invertible; initialise as invertible. If learning rate is low enough it stays invertible
        as long as -logdet is somehow used in the loss, as it will go towards inf when becoming non-invertible.
        
        Logdet of precision is 2 logdet of A, since det(A^T) = det(A) and det(AB) = det(A) det(B),
        this is much more stable than logdet of AA^T
    """
    def forward(self):
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
        """ Function to initialise trainable part, overwrite this to make custom parameters.
        """
        if len(shape)<2:
            raise ValueError(f"Covariance decomposition shape must be at least two dimensions, not {shape}!")
        elif len(shape)==2:
            return torch.eye(*shape, dtype=torch.float32, device=device) * self.init_value
        else:
            prefactor = torch.ones( list(shape)[:-2] + [1,1], dtype=torch.float32, device=device) * self.init_value
            eye = torch.eye( *list(shape)[-2:], dtype=torch.float32, device=device).view( [1]*(len(shape)-2) + list(shape[-2:]) )
            return (prefactor * eye)

class CovarianceTrilExtendableParameter(ExtendableParameter):
    """ ExtendableParameter, but initialised as the covariance decomposition for the rainbow GMM model.
        Directly decomposed as lower triangular matrix to save on lossy conversions.
        
        Only use lower triangular part, pass diagonal through softplus to ensure positivity.
        
        Seems much less stable than PrecisionExtendableParameter, not relly sure why.
    """
    
    def forward(self):
        value = torch.tril( self.param_fixed + self.param_mask_trainable * self.param_trainable )
        value[..., range(value.shape[-2]), range(value.shape[-1])] = F.softplus(value[..., range(value.shape[-2]), range(value.shape[-1])])
        return value
    
    def get_initial_trainable(self, shape, device="cpu"):
        """ Function to initialise trainable part, overwrite this to make custom parameters.
        """
        if len(shape)<2:
            raise ValueError(f"Covariance decomposition shape must be at least two dimensions, not {shape}!")
        elif len(shape)==2:
            return torch.eye(*shape, dtype=torch.float32) * self.init_value
        else:
            prefactor = torch.ones( list(shape)[:-2] + [1,1], dtype=torch.float32) * self.init_value
            eye = torch.eye( *list(shape)[-2:], dtype=torch.float32).view( [1]*(len(shape)-2) + list(shape[-2:]) )
            return (prefactor * eye)
    
    def get_initial_trainable_mask(self, shape, device="cpu"):
        """ Function to initialise trainable mask part, overwrite this to make custom parameters.
        """
        return torch.tril(torch.ones(shape, dtype=torch.float32))

