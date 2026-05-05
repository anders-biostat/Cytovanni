import numpy as np

def symmetric_correlation(x, y):
    """ Symmetric correlation coefficient, i.e. correlation of [x,y] and [y,x].
    """
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    r = np.corrcoef(np.hstack([x,y]), np.hstack([y, x]))[0,1]
    return r

def get_positive_paired_CV(x, y):
    """ Get CV of strictly positive data, appropriate for e.g. (strictly positive) proportions or standard deviations.
        Uses symmetric data, i.e. [x,y] is compared with [y,x].
        Works by going to log scale, calculating the standard deviation around the diagonal, and translating into a CV.
        Returns nan for data <=0

        Test with:
            import scipy.stats
            log_shares = np.linspace(-5,-1, 10000) # range of values
            CV = .6 # set a CV
            std = np.sqrt(np.log(1+CV**2)) # get standard deviation of LogNormal that corresponds to CV
            noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000) # noise
            print("Simple CV", scipy.stats.variation(np.exp(-3+noisex))) # naive CV along only one direction
            print("Paired CV", get_positive_paired_CV(np.exp(log_shares+noisex), np.exp(log_shares+noisey))) # CV using paired measurements along a large range
    """
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    x, y = np.log(x), np.log(y)
    x, y = np.hstack([x, y]), np.hstack([y, x])
    
    unit_tangent = np.array([-1,1])/np.sqrt(2) # by construction for best fit line, since symmetric
    variance = unit_tangent @ np.cov(np.vstack([x, y])) @ unit_tangent # variance orthogonal to diagonal
    CV = np.sqrt(np.exp(variance)-1)
    return CV

def get_positive_paired_CV_alongdiag(x, y):
    """ get_positive_paired_CV, but using variance along the diagonal instead to get the CV of the underlying samples.

        Needs an additional correction for this to work out correctly:
        Because x and y share the same underlying signal, cov(x,y) = σ²_signal, which enters the along-diagonal projection twice,
        giving var_along = 2σ²_signal + σ²_noise. The perpendicular projection cancels the covariance term, giving var_perp = σ²_noise.
        Therefore σ²_signal = (var_along - var_perp) / 2.

        Test with:
            import scipy.stats
            log_shares = np.linspace(-5,-1, 10000) # range of values
            CV = .6 # set a CV
            std = np.sqrt(np.log(1+CV**2)) # get standard deviation of LogNormal that corresponds to CV
            noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000) # noise
            print("True underlying CV", get_positive_single_CV(np.exp(log_shares)))
            print("Estimated underlying CV", get_positive_paired_CV_alongdiag(np.exp(log_shares+noisex), np.exp(log_shares+noisey)))
    """
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
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
        return np.nan
    CV = np.sqrt(np.exp(variance)-1)
    return CV

def get_positive_single_CV(x):
    """ Get CV of strictly positive data, appropriate for e.g. (strictly positive) proportions or standard deviations.
        Works on standard deviation after log for stability, otherwise the same as simple CV.
        Returns nan for data <=0

        Test with:
            import scipy.stats
            CV = .6 # set a CV
            std = np.sqrt(np.log(1+CV**2)) # get standard deviation of LogNormal that corresponds to CV
            noise = np.random.normal(scale=std, size=10000) # noise
            print("Simple CV", scipy.stats.variation(np.exp(noise))) # naive CV
            print("Stable CV", get_positive_single_CV(np.exp(noise))) # more stable CV
    """
    x = np.array(x)
    x = x[np.isfinite(x)]
    x = np.log(x)
    variance = x.var()
    CV = np.sqrt(np.exp(variance)-1) # just std + O(std**2)
    return CV

def get_asinh_paired_CV(x_asinh, y_asinh, f):
    """ Get something like the CV of fluorescence intensities from paired, asinh-transformed (with cofactor f) measurements.
        Uses symmetric data, i.e. [x,y] is compared with [y,x].
        Like get_positive_paired_CV, gets CV from variance orthogonal to diagonal.
        
        Not quite the same as the CV, as this would become non-sensical for low noise, negative values etc.
        Instead effectively takes the standard deviation after asinh-transformation,
        and translates into CV in such a way that it actually is the CV for x>>f.
        For x<f, not actually the CV, but a Gaussian on the asinh-transformed scale that is shifted around yields a roughly constant value,
        that matches the CV if it is shifted to high values.
        
        chatGPT:
        Much simpler global-linearized noise estimator for arcsinh-transformed replicates.
        - linearizes asinh^{-1} at the global mean
        - behaves like multiplicative CV at large values
        - becomes additive-SD / |global_mean| near zero
        - fully signed, never excludes negative values

        Test with:
            import scipy.stats
            import matplotlib.pyplot as plt
            log_shares = np.linspace(1,3, 10000) # range of values
            CV = .5 # set a CV
            std = np.sqrt(np.log(1+CV**2)) # get standard deviation of LogNormal that corresponds to CV
            noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000) # noise
            f, shift = 10, 0
            print("Simple CV of lognormal", scipy.stats.variation(np.exp(noisex)))
            print("Kind-of-CV from asinh ", get_asinh_paired_CV(log_shares+shift+noisex, log_shares+shift+noisey, f))
            plt.hist(log_shares+shift+noisex);
    """
    x = np.asarray(x_asinh)
    y = np.asarray(y_asinh)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
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
    """ get_asinh_paired_CV along the diagonal, with the same correction as in get_positive_paired_CV_alongdiag

        Test with:
            import scipy.stats
            import matplotlib.pyplot as plt
            log_shares = np.linspace(1,2, 10000) # range of values
            CV = .2 # set a CV
            std = np.sqrt(np.log(1+CV**2)) # get standard deviation of LogNormal that corresponds to CV
            noisex, noisey = np.random.normal(scale=std, size=10000), np.random.normal(scale=std, size=10000) # noise
            f, shift = 10, 0
            print("True underlying CV", get_asinh_single_CV(log_shares+shift, f))
            print("Estimated underlying CV", get_asinh_paired_CV_alongdiag(log_shares+shift+noisex, log_shares+shift+noisey, f))
    """
    x = np.asarray(x_asinh)
    y = np.asarray(y_asinh)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
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
        return np.nan
    
    # global mean and derivative
    m_global = np.mean(s1)
    z0 = f * np.sinh(m_global)
    dzds = np.sqrt(z0**2 + f**2)
    
    # delta-method (no /2 needed now)
    std_z = dzds * np.sqrt(var_s)
    
    denom = max(abs(z0), f)
    return std_z / denom

def get_asinh_single_CV(x_asinh, f):
    """ Same as get_asinh_paired_CV, but using only one x and getting variation around mean.
    
        Test with:
            import scipy.stats
            import matplotlib.pyplot as plt
            CV = .3 # set a CV
            std = np.sqrt(np.log(1+CV**2)) # get standard deviation of LogNormal that corresponds to CV
            noisex = np.random.normal(scale=std, size=10000) # noise
            f, shift = 10, 2
            print("Simple CV of lognormal", scipy.stats.variation(np.exp(noisex)))
            print("Kind-of-CV from asinh ", get_asinh_single_CV(shift+noisex, f))
            plt.hist(shift+noisex);
    """

    x = np.asarray(x_asinh)
    x = x[np.isfinite(x)]

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
    """ Same as get_asinh_paired_CV, but with additional correction.
        If both x and y have inherent error, subtract this variance from observed to get corrected values.
    """
    x = np.asarray(x_asinh)
    y = np.asarray(y_asinh)
    x_err = np.asarray(x_asinh_err)
    y_err = np.asarray(y_asinh_err)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_err) & np.isfinite(y_err) & (x_err>0) & (y_err>0)
    x, y, x_err, y_err = x[mask], y[mask], x_err[mask], y_err[mask]
    
    # symmetrical pairing
    s1 = np.hstack([x, y])
    s2 = np.hstack([y, x])
    
    # covariance approach
    unit_tangent = np.array([-1, 1]) / np.sqrt(2)
    var_s = unit_tangent @ np.cov(np.vstack([s1, s2])) @ unit_tangent

    # error variance
    var_s_err = .5 * np.mean(x_err**2 + y_err**2)
    var_s_corrected = max(var_s - var_s_err, 0)
    
    # global mean and derivative
    m_global = np.mean(s1)
    z0 = f * np.sinh(m_global)
    dzds = np.sqrt(z0**2 + f**2)
    
    # delta-method (no /2 needed now)
    std_z = dzds * np.sqrt(var_s_corrected)
    
    denom = max(abs(z0), f)
    return std_z / denom

def get_asinh_single_CV_varcorr(x_asinh, x_asinh_err, f):
    """ get_asinh_single_CV with variance correction as in get_asinh_paired_CV_varcorr
    """

    x = np.asarray(x_asinh)
    x_err = np.asarray(x_asinh_err)
    mask = np.isfinite(x) & np.isfinite(x_err) & (x_err>0)
    x, x_err = x[mask], x_err[mask]

    # global mean in asinh space
    m_global = np.mean(x)

    # latent scale point
    z0 = f * np.sinh(m_global)

    # global derivative
    dzds = np.sqrt(z0**2 + f**2)

    # variance in asinh space
    var_s = x.var(ddof=1)
    var_s_corrected = var_s - np.mean(x_err**2)

    # delta-method
    var_z = (dzds**2) * var_s_corrected

    # latent std
    std_z = np.sqrt(var_z)

    # denominator identical to replicate version
    denom = max(abs(z0), f)

    return std_z / denom


