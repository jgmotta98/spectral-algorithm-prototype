import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


def WhittakerSmooth(x: list[float], w: np.ndarray[float], 
                    lambda_: int, differences: int = 1) -> np.ndarray:
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


def airPLS(x: list[float], lambda_: int = 100, 
           porder: int = 1, itermax: int = 15) -> np.ndarray:
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


def input_baseline_correction(spectra_input: list[str, pd.DataFrame]) -> pd.DataFrame:
    spectra_input[1].insert(0, 'name', spectra_input[0])
    spectra_input = spectra_input[1]
    
    spectra_input['y'] = 2 - np.log10(spectra_input['y'])
    c1=np.array(spectra_input['y'].tolist())-airPLS(np.array(spectra_input['y'].tolist()))
    
    c1 = 10**(-c1) * 100
    spectra_input['y'] = c1.tolist()

    return spectra_input
