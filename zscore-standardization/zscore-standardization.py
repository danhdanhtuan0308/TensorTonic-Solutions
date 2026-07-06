import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    safe_X = np.array(X ) 
    z =  ( safe_X - np.mean(safe_X, axis = axis , keepdims = True))/ ( np.std(safe_X,axis = axis , keepdims = True) + eps ) 
    return z 