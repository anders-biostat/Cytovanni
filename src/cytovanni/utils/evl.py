"""Evaluation metrics for paired measurements: coefficients of variation (CV) and a symmetric correlation.

Paired measurements are two measurements of the same quantity, e.g. a cell type proportion or a mean fluorescence
intensity (MFI) of the same sample measured in two batches. All paired metrics are symmetrised: every pair (x, y) is
also used as (y, x), so the best-fit line is the diagonal by construction and neither batch is treated as the
reference. The variance orthogonal to the diagonal measures the disagreement of the pairs (measurement error); the
`_alongdiag` variants instead estimate the spread of the underlying values without the measurement error.

Two families:

- ``get_positive_*``: strictly positive quantities (proportions, standard deviations), assumed log-normal; the CV is
  computed from the variance of the logarithm.
- ``get_asinh_*``: arcsinh-transformed intensities ``arcsinh(z / f)``; a CV-like value that is the CV for intensities
  well above the cofactor f and the standard deviation divided by f below it.

All functions return the CV as a fraction (not in percent) and NaN, with a `CytovanniWarning`, if too few usable
values remain.
"""
import numpy as np
import warnings

from ..exceptions import CytovanniWarning

def _nan_with_warning(fname, reason):
    """ Return NaN with a CytovanniWarning naming the function and the reason. """
    warnings.warn(f"{fname}: {reason}, returning NaN.", CytovanniWarning, stacklevel=3)
    return np.nan

def _positive_pairs(x, y):
    """ Pairs where both values are finite and > 0 (the log is undefined otherwise). """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    return x[mask], y[mask]

def symmetric_correlation(x, y):
    """Pearson correlation of the symmetrised pairs, i.e. of ``[x, y]`` with ``[y, x]``.

    Unlike the plain Pearson correlation, this also decreases when the pairs are shifted or scaled relative to each
    other (it measures agreement with the diagonal, not only linear association).

    Parameters
    ----------
    x, y : array-like of float
        Paired values, same length. Pairs with a non-finite value are dropped.

    Returns
    -------
    float
        Correlation coefficient.
    """
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    r = np.corrcoef(np.hstack([x,y]), np.hstack([y, x]))[0,1]
    return r

def get_positive_paired_CV(x, y):
    """CV of the disagreement of paired, strictly positive measurements (e.g. proportions).

    Takes the logarithm, symmetrises the pairs, and converts the variance orthogonal to the diagonal into the CV of a
    log-normal distribution. The result is the expected CV of a single measurement, e.g. of a proportion measured in
    different batches.

    Parameters
    ----------
    x, y : array-like of float
        Paired measurements on the linear scale, same length. Pairs with a value <= 0 or a non-finite value are
        dropped.

    Returns
    -------
    float
        CV as a fraction. NaN (with a `CytovanniWarning`) if fewer than 2 pairs remain.

    See Also
    --------
    get_positive_paired_CV_alongdiag : CV of the underlying values instead of the measurement error.
    get_asinh_paired_CV : for arcsinh-transformed intensities.

    Notes
    -----
    With the covariance matrix Sigma of the symmetrised log values and ``u = [-1, 1] / sqrt(2)``, the variance is
    ``sigma2 = u @ Sigma @ u`` and the CV ``sqrt(exp(sigma2) - 1)``.

    Examples
    --------
    >>> log_shares = np.linspace(-5, -1, 10000)  # range of true log proportions
    >>> CV = .6
    >>> std = np.sqrt(np.log(1 + CV**2))  # log-normal standard deviation for this CV
    >>> noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000)
    >>> get_positive_paired_CV(np.exp(log_shares + noisex), np.exp(log_shares + noisey))  # about 0.6
    """
    x, y = _positive_pairs(x, y)
    if len(x) < 2:
        return _nan_with_warning("get_positive_paired_CV", f"{len(x)} usable pairs (finite, > 0)")
    x, y = np.log(x), np.log(y)
    x, y = np.hstack([x, y]), np.hstack([y, x])
    
    unit_tangent = np.array([-1,1])/np.sqrt(2) # by construction for best fit line, since symmetric
    variance = unit_tangent @ np.cov(np.vstack([x, y])) @ unit_tangent # variance orthogonal to diagonal
    CV = np.sqrt(np.exp(variance)-1)
    return CV

def get_positive_paired_CV_alongdiag(x, y):
    """CV of the underlying values of paired, strictly positive measurements, without the measurement error.

    Like `get_positive_paired_CV`, but estimates the spread of the true values (e.g. how much a proportion varies
    between samples) from the variance along the diagonal, corrected for the measurement error.

    Parameters
    ----------
    x, y : array-like of float
        Paired measurements on the linear scale, same length. Pairs with a value <= 0 or a non-finite value are
        dropped.

    Returns
    -------
    float
        CV as a fraction. NaN (with a `CytovanniWarning`) if fewer than 2 pairs remain, or if the variance along the
        diagonal is smaller than the orthogonal one (the measurement error exceeds the spread of the values).

    See Also
    --------
    get_positive_paired_CV : CV of the measurement error.

    Notes
    -----
    With ``x = s + e_x`` and ``y = s + e_y`` on the log scale, the covariance of x and y is the variance of the
    signal s. Projecting the covariance matrix of the symmetrised pairs onto ``[1, 1] / sqrt(2)`` gives
    ``var_along = 2 var(s) + var(e)``, onto ``[-1, 1] / sqrt(2)`` gives ``var_perp = var(e)``, so
    ``var(s) = (var_along - var_perp) / 2``, converted to a CV as in `get_positive_paired_CV`.

    Examples
    --------
    >>> log_shares = np.linspace(-5, -1, 10000)
    >>> CV = .6
    >>> std = np.sqrt(np.log(1 + CV**2))
    >>> noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000)
    >>> get_positive_single_CV(np.exp(log_shares))  # true CV of the underlying values, about 1.67
    >>> get_positive_paired_CV_alongdiag(np.exp(log_shares + noisex), np.exp(log_shares + noisey))  # about 1.67
    """
    x, y = _positive_pairs(x, y)
    if len(x) < 2:
        return _nan_with_warning("get_positive_paired_CV_alongdiag", f"{len(x)} usable pairs (finite, > 0)")
    x, y = np.log(x), np.log(y)
    x, y = np.hstack([x, y]), np.hstack([y, x])

    # orthogonal
    unit_tangent = np.array([-1,1])/np.sqrt(2) # by construction for best fit line, since symmetric
    variance_ortho = unit_tangent @ np.cov(np.vstack([x, y])) @ unit_tangent # variance orthogonal to diagonal
    # along diagonal
    unit_tangent = np.array([1,1])/np.sqrt(2) # by construction for best fit line, since symmetric
    variance_along = unit_tangent @ np.cov(np.vstack([x, y])) @ unit_tangent # variance along diagonal
    # combined
    variance = (variance_along - variance_ortho)/2
    if variance<0:
        return _nan_with_warning("get_positive_paired_CV_alongdiag", f"variance along the diagonal is below the orthogonal one ({variance:.3g} < 0)")
    CV = np.sqrt(np.exp(variance)-1)
    return CV

def get_positive_single_CV(x):
    """CV of strictly positive values (e.g. proportions), computed on the log scale.

    Converts the sample variance (ddof=1) of the logarithm into the CV of a log-normal distribution. For log-normal
    data this is more stable than the plain CV (standard deviation / mean), which is sensitive to the tail.

    Parameters
    ----------
    x : array-like of float
        Values on the linear scale. Values <= 0 or non-finite are dropped.

    Returns
    -------
    float
        CV as a fraction, ``sqrt(exp(var(log x)) - 1)``. NaN (with a `CytovanniWarning`) if fewer than 2 values
        remain.

    Examples
    --------
    >>> CV = .6
    >>> std = np.sqrt(np.log(1 + CV**2))
    >>> get_positive_single_CV(np.exp(np.random.normal(scale=std, size=10000)))  # about 0.6
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x>0)]
    if len(x) < 2:
        return _nan_with_warning("get_positive_single_CV", f"{len(x)} usable values (finite, > 0)")
    x = np.log(x)
    variance = x.var(ddof=1)
    CV = np.sqrt(np.exp(variance)-1) # just std + O(std**2)
    return CV

def get_asinh_paired_CV(x_asinh, y_asinh, f):
    """CV-like measure of the disagreement of paired arcsinh-transformed intensities (e.g. MFIs).

    The arcsinh analogue of `get_positive_paired_CV`: takes the variance of the symmetrised pairs orthogonal to the
    diagonal on the arcsinh scale and translates it to the linear scale with the delta method. For intensities well
    above the cofactor the result is the CV of a single measurement; below the cofactor, where a relative CV is not
    meaningful (values near zero or negative), it is the additive standard deviation divided by `f`. So for the same
    relative noise, the value grows with the intensity as long as it is below `f`, and depends on the choice of `f`
    there. Negative values are used like all others.

    Parameters
    ----------
    x_asinh, y_asinh : array-like of float
        Paired measurements on the arcsinh scale, ``arcsinh(z / f)``, same length. Pairs with a non-finite value
        are dropped.
    f : float
        Cofactor of the arcsinh transform of `x_asinh` and `y_asinh`, in the units of the linear data z.

    Returns
    -------
    float
        CV as a fraction. NaN (with a `CytovanniWarning`) if fewer than 2 pairs remain.

    See Also
    --------
    get_asinh_paired_CV_alongdiag : spread of the underlying values instead of the measurement error.
    get_asinh_paired_CV_varcorr : with a correction for known errors of the individual values.
    get_asinh_single_CV : for unpaired values.

    Notes
    -----
    With the variance ``var_s`` orthogonal to the diagonal (as in `get_positive_paired_CV`, but on the arcsinh
    scale) and the pooled mean m of all values, the linear-scale point is ``z0 = f * sinh(m)``, the derivative of the
    inverse transform there ``sqrt(z0**2 + f**2)``, and the result
    ``sqrt(z0**2 + f**2) * sqrt(var_s) / max(|z0|, f)``.

    The linearisation uses one point, the pooled mean, for all pairs. The factor ``sqrt(z0**2 + f**2) / max(|z0|, f)``
    lies in [1, sqrt(2)], so this changes the result by at most a factor sqrt(2) compared to linearising every pair
    separately (worst case: pooled mean near f while the pairs lie far above or below it); for pooled means far above
    or below f it is close to 1.

    Examples
    --------
    >>> log_shares = np.linspace(1, 3, 10000)  # arcsinh values, i.e. z between about 1.2 f and 10 f
    >>> CV = .5
    >>> std = np.sqrt(np.log(1 + CV**2))
    >>> noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000)
    >>> get_asinh_paired_CV(log_shares + noisex, log_shares + noisey, f=10)  # roughly 0.5
    """
    x = np.asarray(x_asinh)
    y = np.asarray(y_asinh)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return _nan_with_warning("get_asinh_paired_CV", f"{len(x)} finite pairs")
    
    # symmetrical pairing
    s1 = np.hstack([x, y])
    s2 = np.hstack([y, x])
    
    # covariance approach
    unit_tangent = np.array([-1, 1]) / np.sqrt(2)
    var_s = unit_tangent @ np.cov(np.vstack([s1, s2])) @ unit_tangent
    
    # global mean and derivative
    m_global = np.mean(s1)
    z0 = f * np.sinh(m_global)
    dzds = np.sqrt(z0**2 + f**2)
    
    # delta-method (no /2 needed now)
    std_z = dzds * np.sqrt(var_s)
    
    denom = max(abs(z0), f)
    return std_z / denom

def get_asinh_paired_CV_alongdiag(x_asinh, y_asinh, f):
    """CV-like measure of the spread of the underlying values of paired arcsinh-transformed intensities.

    The arcsinh analogue of `get_positive_paired_CV_alongdiag`: the variance of the underlying values is estimated as
    ``(var_along - var_perp) / 2`` on the arcsinh scale and translated to the linear scale as in
    `get_asinh_paired_CV` (same cofactor behaviour and global linearisation, see there).

    Parameters
    ----------
    x_asinh, y_asinh : array-like of float
        Paired measurements on the arcsinh scale, ``arcsinh(z / f)``, same length. Pairs with a non-finite value
        are dropped.
    f : float
        Cofactor of the arcsinh transform, in the units of the linear data z.

    Returns
    -------
    float
        CV as a fraction. NaN (with a `CytovanniWarning`) if fewer than 2 pairs remain, or if the variance along the
        diagonal is smaller than the orthogonal one.

    Examples
    --------
    >>> log_shares = np.linspace(1, 2, 10000)
    >>> CV = .2
    >>> std = np.sqrt(np.log(1 + CV**2))
    >>> noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000)
    >>> get_asinh_single_CV(log_shares, f=10)  # spread of the underlying values
    >>> get_asinh_paired_CV_alongdiag(log_shares + noisex, log_shares + noisey, f=10)  # close to the line above
    """
    x = np.asarray(x_asinh)
    y = np.asarray(y_asinh)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return _nan_with_warning("get_asinh_paired_CV_alongdiag", f"{len(x)} finite pairs")
    
    # symmetrical pairing
    s1 = np.hstack([x, y])
    s2 = np.hstack([y, x])
    
    # orthogonal
    unit_tangent = np.array([-1,1])/np.sqrt(2) # by construction for best fit line, since symmetric
    variance_ortho = unit_tangent @ np.cov(np.vstack([s1, s2])) @ unit_tangent # variance orthogonal to diagonal
    # along diagonal
    unit_tangent = np.array([1,1])/np.sqrt(2) # by construction for best fit line, since symmetric
    variance_along = unit_tangent @ np.cov(np.vstack([s1, s2])) @ unit_tangent # variance along diagonal
    # combined
    var_s = (variance_along - variance_ortho)/2

    if var_s<0:
        return _nan_with_warning("get_asinh_paired_CV_alongdiag", f"variance along the diagonal is below the orthogonal one ({var_s:.3g} < 0)")
    
    # global mean and derivative
    m_global = np.mean(s1)
    z0 = f * np.sinh(m_global)
    dzds = np.sqrt(z0**2 + f**2)
    
    # delta-method (no /2 needed now)
    std_z = dzds * np.sqrt(var_s)
    
    denom = max(abs(z0), f)
    return std_z / denom

def get_asinh_single_CV(x_asinh, f):
    """CV-like measure of the spread of unpaired arcsinh-transformed intensities around their mean.

    The single-sample version of `get_asinh_paired_CV`: the sample variance (ddof=1) on the arcsinh scale is
    translated to the linear scale in the same way (same cofactor behaviour and global linearisation, see there).

    Parameters
    ----------
    x_asinh : array-like of float
        Values on the arcsinh scale, ``arcsinh(z / f)``. Non-finite values are dropped.
    f : float
        Cofactor of the arcsinh transform, in the units of the linear data z.

    Returns
    -------
    float
        CV as a fraction. NaN (with a `CytovanniWarning`) if fewer than 2 values remain.

    Examples
    --------
    >>> CV = .3
    >>> std = np.sqrt(np.log(1 + CV**2))
    >>> noisex = np.random.normal(scale=std, size=10000)
    >>> get_asinh_single_CV(2 + noisex, f=10)  # z around 36, well above f: roughly 0.3
    """

    x = np.asarray(x_asinh)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return _nan_with_warning("get_asinh_single_CV", f"{len(x)} finite values")

    # global mean in asinh space
    m_global = np.mean(x)

    # latent scale point
    z0 = f * np.sinh(m_global)

    # global derivative
    dzds = np.sqrt(z0**2 + f**2)

    # variance in asinh space
    var_s = x.var(ddof=1)

    # delta-method
    var_z = (dzds**2) * var_s

    # latent std
    std_z = np.sqrt(var_z)

    # denominator identical to replicate version
    denom = max(abs(z0), f)

    return std_z / denom


def get_asinh_paired_CV_varcorr(x_asinh, y_asinh, x_asinh_err, y_asinh_err, f):
    """`get_asinh_paired_CV`, corrected for known errors of the individual values.

    Subtracts the mean error variance of the values from the observed variance orthogonal to the diagonal before the
    translation to the linear scale, e.g. to remove the sampling error of MFIs of small populations, estimated by a
    bootstrap.

    Parameters
    ----------
    x_asinh, y_asinh : array-like of float
        Paired measurements on the arcsinh scale, ``arcsinh(z / f)``, same length.
    x_asinh_err, y_asinh_err : array-like of float
        Standard error of every value of `x_asinh` and `y_asinh`, on the arcsinh scale. Must be finite and >= 0 (an
        error of 0 subtracts nothing). Pairs with a non-finite value or a negative or non-finite error are dropped.
    f : float
        Cofactor of the arcsinh transform, in the units of the linear data z.

    Returns
    -------
    float
        CV as a fraction. If the mean error variance ``mean(x_err**2 + y_err**2) / 2`` exceeds the observed
        variance, the corrected variance is clamped to 0 (with a `CytovanniWarning`), so the result is 0. NaN (with
        a `CytovanniWarning`) if fewer than 2 usable pairs remain.
    """
    x = np.asarray(x_asinh)
    y = np.asarray(y_asinh)
    x_err = np.asarray(x_asinh_err)
    y_err = np.asarray(y_asinh_err)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_err) & np.isfinite(y_err) & (x_err>=0) & (y_err>=0)
    x, y, x_err, y_err = x[mask], y[mask], x_err[mask], y_err[mask]
    if len(x) < 2:
        return _nan_with_warning("get_asinh_paired_CV_varcorr", f"{len(x)} usable pairs (finite, error >= 0)")
    
    # symmetrical pairing
    s1 = np.hstack([x, y])
    s2 = np.hstack([y, x])
    
    # covariance approach
    unit_tangent = np.array([-1, 1]) / np.sqrt(2)
    var_s = unit_tangent @ np.cov(np.vstack([s1, s2])) @ unit_tangent

    # error variance
    var_s_err = .5 * np.mean(x_err**2 + y_err**2)
    var_s_corrected = var_s - var_s_err
    if var_s_corrected < 0: # error variance exceeds the observed variance: no measurable spread
        warnings.warn(f"get_asinh_paired_CV_varcorr: corrected variance is negative ({var_s_corrected:.3g}), clamped to 0 (CV 0).", CytovanniWarning)
        var_s_corrected = 0.
    
    # global mean and derivative
    m_global = np.mean(s1)
    z0 = f * np.sinh(m_global)
    dzds = np.sqrt(z0**2 + f**2)
    
    # delta-method (no /2 needed now)
    std_z = dzds * np.sqrt(var_s_corrected)
    
    denom = max(abs(z0), f)
    return std_z / denom

def get_asinh_single_CV_varcorr(x_asinh, x_asinh_err, f):
    """`get_asinh_single_CV`, corrected for known errors of the individual values.

    Subtracts the mean error variance ``mean(x_err**2)`` from the sample variance on the arcsinh scale before the
    translation to the linear scale, as in `get_asinh_paired_CV_varcorr`.

    Parameters
    ----------
    x_asinh : array-like of float
        Values on the arcsinh scale, ``arcsinh(z / f)``.
    x_asinh_err : array-like of float
        Standard error of every value, on the arcsinh scale. Must be finite and >= 0 (an error of 0 subtracts
        nothing). Values with a non-finite value or a negative or non-finite error are dropped.
    f : float
        Cofactor of the arcsinh transform, in the units of the linear data z.

    Returns
    -------
    float
        CV as a fraction. If the mean error variance exceeds the observed variance, the corrected variance is
        clamped to 0 (with a `CytovanniWarning`), so the result is 0. NaN (with a `CytovanniWarning`) if fewer than 2
        usable values remain.
    """

    x = np.asarray(x_asinh)
    x_err = np.asarray(x_asinh_err)
    mask = np.isfinite(x) & np.isfinite(x_err) & (x_err>=0)
    x, x_err = x[mask], x_err[mask]
    if len(x) < 2:
        return _nan_with_warning("get_asinh_single_CV_varcorr", f"{len(x)} usable values (finite, error >= 0)")

    # global mean in asinh space
    m_global = np.mean(x)

    # latent scale point
    z0 = f * np.sinh(m_global)

    # global derivative
    dzds = np.sqrt(z0**2 + f**2)

    # variance in asinh space
    var_s = x.var(ddof=1)
    var_s_corrected = var_s - np.mean(x_err**2)
    if var_s_corrected < 0: # error variance exceeds the observed variance: no measurable spread
        warnings.warn(f"get_asinh_single_CV_varcorr: corrected variance is negative ({var_s_corrected:.3g}), clamped to 0 (CV 0).", CytovanniWarning)
        var_s_corrected = 0.

    # delta-method
    var_z = (dzds**2) * var_s_corrected

    # latent std
    std_z = np.sqrt(var_z)

    # denominator identical to replicate version
    denom = max(abs(z0), f)

    return std_z / denom


