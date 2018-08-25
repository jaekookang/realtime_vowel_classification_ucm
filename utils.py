'''
This code includes functions to help UCM analysis

Jaekoo Kang
'''
import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from numpy.linalg import svd
import pdb


def read_mat(matfile):
    '''Read matfile'''
    FileID = re.sub('.mat', '', os.path.basename(matfile))
    return loadmat(matfile)[FileID]

def find_elements(pattern, my_list):
    '''Find elements in a list'''
    elements = []
    index = []

    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            elements.append(my_list[i])
            index.append(i)
    return index, elements


def get_palate(mat, which_spkr):
    '''Select (pal, pha) for which_spkr'''
    pal = [m[1] for m in mat if m[0][0] == which_spkr][0]
    pha = [m[2] for m in mat if m[0][0] == which_spkr][0]
    return pal, pha    


def remove_short_tokens(df, which_spkr, threshold=0.03):
    '''
    Remove short tokens (default < 30ms)
    '''
    n_samples = df.loc[df.Subj==which_spkr].shape[0]
    tmp = df.loc[(df.Subj==which_spkr)&(df.Dur<threshold)]
    n_trimmed = tmp.shape[0]
    df.drop(tmp.index, inplace=True)
    print(f'{which_spkr}: {n_samples} --> {n_samples-n_trimmed} ({n_trimmed})')
    return df


def iqr_bounds(df, col, lb=0.25, ub=0.75):
    '''
    IQR covers (ub-lb)*100 % of data from the median
      lb: lower bound (default=0.25)
      ub: upper bound (default=0.75)
    '''
    q1 = df[col].quantile(lb)
    q3 = df[col].quantile(ub)
    IQR = q3 - q1
    lrbound = q1 - 1.5*IQR
    upbound = q3 + 1.5*IQR
    return lrbound, upbound

def remove_acous_outlier(df, which_spkr, which_vowel, lower_bound=0.25, upper_bound=0.75):
    '''
    Remove outliers in acoustics data (F1, F2, F3)

    lb: lower bound
    ub: upper bound
    '''
    d = df.loc[(df.Subj==which_spkr)&(df.Label==which_vowel)]
    # F1
    lb, ub = iqr_bounds(d, 'F1', lb=lower_bound, ub=upper_bound)
    f1outLier = df.loc[(df.Subj==which_spkr)&(df.Label==which_vowel)&((df.F1<lb)|(df.F1>ub))]
    df.drop(f1outLier.index, inplace=True) # no need to reassign
    # F2
    lb, ub = iqr_bounds(d, 'F2', lb=lower_bound, ub=upper_bound)
    f2outLier = df.loc[(df.Subj==which_spkr)&(df.Label==which_vowel)&((df.F2<lb)|(df.F2>ub))]
    df.drop(f2outLier.index, inplace=True) # no need to reassign
    # F3
    lb, ub = iqr_bounds(d, 'F3', lb=lower_bound, ub=upper_bound)
    f3outLier = df.loc[(df.Subj==which_spkr)&(df.Label==which_vowel)&((df.F3<lb)|(df.F3>ub))]
    df.drop(f3outLier.index, inplace=True) # no need to reassign
    return df, (f1outLier, f2outLier, f3outLier)


def normalize_formants(formants, mu=None, sigma=None):
    '''
    This function normalizes (z-score) formant frequencies and outputs the normalized formants.
    You can later append it to the original data frame.

    If (mu, sigma) is provided, formants will be normalized with these values.
    
    Arguments
      formants: formant frequencies; e.g., N x 3 numpy array
      (optionally) mu: mean value, sigma: standard deviation to use
    
    Output
      normalized: normalized formants; e.g., N x 3 numpy array
      mu: mean
      sigma: standard deviation
    '''
    # Check if NaNs
    if np.isnan(formants).any():
        print('WARNING: NaN values were found')
        
    if not (mu is None and sigma is None):
        return (formants - mu) / sigma, mu, sigma
    else:
        mu = np.nanmean(formants, axis=0, keepdims=True) # ignore NaNs
        sigma = np.nanstd(formants, axis=0, keepdims=True) # ignore NaNs
        return (formants - mu) / sigma, mu, sigma


def normalize_pellets(pellets):
    '''
    This function normalizes (z-score) pellet xy coordinates and outputs the normalized pellets.
    You can later append it to the original data frame.
    
    Arguments
      pellets: pellet xy coordinates; e.g., N x 12 numpy array
      
    Output
      normalized: normalized pellets; e.g., N x 12 numpy array
      mu: mean
      sigma: standard deviation
    '''
    if np.isnan(pellets).any():
        print('WARNING: NaN values were found')
    mu = np.nanmean(pellets, axis=0, keepdims=True) # ignore NaNs
    sigma = np.nanstd(pellets, axis=0, keepdims=True) # ignore NaNs
    return (pellets - mu) / sigma, mu, sigma


def run_pca(norm_pellets, n_comp=5, apply_factor_analysis=False):
    '''
    This function runs PCA on normalized pellet data and returns pca object
    
    apply_factor_analysis will be considered for the factor analysis (TODO)
    '''
    pca = PCA(n_components=n_comp)
    pca.fit(norm_pellets)
    return pca


def zscore(data, mu, sd):
    return (data - mu) / sd


def zscore_reverse(zdata, mu, sigma):
    '''
    This function returns raw values from zscored data with mu and sigma
    '''
    return zdata*sigma + mu


def calc_rmse(y, yhat):
    '''
    Calculate a root mean-squared error
    
    rmse = 1/n sqrt( 1/m * (y - yhat)^2 )
    '''
    return np.mean(np.sqrt(np.mean(np.square(y-yhat), axis=1)))


def zscore_reverse_all(data, formant_zscore, which_spkr):
    '''
    data: zscored formants, Nx3
    formant_zscore: parameters for zscore (mu, sd)
    '''
    F1 = zscore_reverse(data[:, 0], formant_zscore[which_spkr]['mu'][0][0],
                        formant_zscore[which_spkr]['sigma'][0][0])
    F2 = zscore_reverse(data[:, 1], formant_zscore[which_spkr]['mu'][0][1],
                        formant_zscore[which_spkr]['sigma'][0][1])
    F3 = zscore_reverse(data[:, 2], formant_zscore[which_spkr]['mu'][0][2],
                        formant_zscore[which_spkr]['sigma'][0][2])
    return np.array([F1, F2, F3]).T


def get_rmse(y, yhat):
    '''Calculate RMSE (Hz)'''
    if len(y.shape) == 1:
        return np.sqrt(np.multiply(y - yhat, y - yhat) / len(y))
    else:
        return np.sqrt(np.mean(np.multiply(y - yhat, y - yhat), axis=0))


def get_mean_absolute_error(y, yhat):
    '''Calculate MSE (Hz)'''
    if len(y.shape) == 1:
        return np.mean(np.absolute(y - yhat))
    else:
        return np.mean(np.absolute(y - yhat), axis=0)

    
def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns