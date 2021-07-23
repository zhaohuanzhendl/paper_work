import numpy as np


def center_kernel(K, copy=True):
    '''
    Centered version of a kernel matrix (corresponding to centering the)
    implicit feature map.
    '''
    means = K.mean(axis=0)
    if copy:
        K = K - means[None, :]
    else:
        K -= means[None, :]
    K -= means[:, None]
    K += means.mean()
    return K


def kat(K1, K2):
    '''
    return the kernel alignment
    '''
    #kta = <K1, K2>_F / (||K1||_F ||K2||_F)
    kta =  np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)
    return kat
