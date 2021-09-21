"""
Two-sample hypothesis test with Maximum Mean Discrepancy (MMD) method.
ref: https://github.com/afarahi/tatter
"""

from __future__ import division
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import pairwise_kernels
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def MMD2u_estimator(K, m, n):
    """
    Compute the MMD^2_u unbiased statistic.
    This is an implementation of an unbiased MMD^2_u estimator:
    Equation (3) in Gretton et al. Journal of Machine Learning Research 13 (2012) 723-773

    :param K: numpy-array
        the pair-wise kernel matrix
    :param m: int
        dimension of the first data set
    :param n: int
        dimension of the second data set

    :return: float
        an unbiased estimate of MMD^2_u
    """
    K_x = K[:m, :m]
    K_y = K[m:, m:]
    K_xy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (K_x.sum() - K_x.diagonal().sum()) + \
            1.0 / (n * (n - 1.0)) * (K_y.sum() - K_y.diagonal().sum()) - \
            2.0 / (m * n) * K_xy.sum()


def MMD_null_estimator(Kstacked, m, n, rng, **kwargs):
    """
    Compute the MMD^2_u for one bootstrap realization.

    :param Kstacked: numpy-array
        stacked X and Y data vector
    :param m: int
        dimension of the first data set
    :param n: int
        dimension of the second data set
    :param rng: type(np.random.RandomState())
        a numpy random function

    :return: float
        an unbiased estimate of MMD^2_u
    """
    idx = rng.permutation(m + n)
    return MMD2u_estimator(Kstacked[idx, idx[:, None]], m, n)



def compute_null_distribution(K, m, n, model='MMD', iterations=1000, n_jobs=1,
                              verbose=False, random_state=None, **kwargs):
    """
    Compute the null-distribution of test statistics via a bootstrap algorithm.

    :param K: numpy-array
        the pair-wise kernel matrix, or stacked data sets X and Y
    :param m: int
        dimension of the first data set
    :param n: int
        dimension of the second data set
    :param model: string
        defines the basis model to perform two sample test ['KS', 'KL', 'MMD']
    :param iterations: int
        controls the number of bootstrap realizations
    :param verbose: bool
        controls the verbosity of the model's output.
    :param random_state: type(np.random.RandomState()) or None
        defines the initial random state.
    :param kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters or `KL_divergence_estimator()`
         as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    :return: numpy-array
        the null distribution
    """

    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    test_null = np.zeros(iterations)

    if verbose:
        iterations_list = tqdm(range(iterations))
    else:
        iterations_list = range(iterations)

    # compute the null distribution

    # for 1 cpu run the normal code, for more cpu use the Parallel library. This maximize the speed.
    if n_jobs == 1:

        if model == 'MMD':
            for i in iterations_list:
                idx = rng.permutation(m+n)
                K_i = K[idx, idx[:, None]]
                test_null[i] = MMD2u_estimator(K_i, m, n)

    else:
        if model == 'MMD':
            null_estimator = MMD_null_estimator

        test_null = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(null_estimator)(K, m, n, rng, **kwargs)
                                                              for _ in iterations_list)

    return test_null


def two_sample_test(X, Y, model='MMD', kernel_function='rbf', iterations=1000, verbose=False,
                    random_state=None, n_jobs=1, **kwargs):
    """
     This function performs two sample test. The two-sample hypothesis test is concerned with whether distributions $p$
      and $q$ are different on the basis of finite samples drawn from each of them. This ubiquitous problem appears
      in a legion of applications, ranging from data mining to data analysis and inference. This implementation can
      perform the Kolmogorov-Smirnov test (for one-dimensional data only), Kullback-Leibler divergence and MMD.
      The module perform a bootstrap algorithm to estimate the null distribution, and corresponding p-value.

    :param X: numpy-array
        Data, of size MxD [M is the number of data points, D is the features dimension]
    :param Y: numpy-array
        Data, of size NxD [N is the number of data points, D is the features dimension]
    :param model: string
        defines the basis model to perform two sample test ['KS', 'KL', 'MMD']
    :param kernel_function: string
        defines the kernel function, only used for the MMD.
        For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
    :param iterations: int
        controls the number of bootstrap realizations
    :param verbose: bool
        controls the verbosity of the model's output.
    :param random_state: type(np.random.RandomState()) or None
        defines the initial random state.
    :param n_jobs: int
        number of jobs to run in parallel.
    :param kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters or `KL_divergence_estimator()`
         as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    :return: tuple of size 3
        float: the test value,
        numpy-array: a null distribution via bootstraps,
        float: estimated p-value
    """

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices. X and Y should have the same feature dimension,"
                         ": X.shape[1] == %i while Y.shape[1] == %i."%(X.shape[1], Y.shape[1]))

    if not (isinstance(iterations, int) and iterations > 1):
        raise ValueError('iterations has incorrect type or less than 2.')

    if not (isinstance(n_jobs, int) and n_jobs > 0):
        raise ValueError('n_jobs is incorrect type or <1. n_jobs:%s'%n_jobs)

    m = len(X)
    n = len(Y)

    # p-value's resolution
    resolution = 1.0/iterations

    # compute the test statistics according to the input model
    if model == 'MMD':
        XY = np.vstack([X, Y])
        K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
        test_value = MMD2u_estimator(K, m, n)

    if verbose:
        print("test value = %s"%test_value)
        print("Computing the null distribution.")

    # compute the null distribution via a bootstrap algorithm
    test_null = compute_null_distribution(K, m, n, model=model, iterations=iterations, verbose=verbose,
                                          n_jobs=n_jobs, random_state=random_state, **kwargs)

    # compute the p-value, if less then the resolution set it to the resolution
    p_value = max(resolution, resolution*(test_null > test_value).sum())

    if verbose:
        if p_value == resolution:
            print("p-value < %s \t (resolution : %s)" % (p_value, resolution))
        else:
            print("p-value ~= %s \t (resolution : %s)" % (p_value, resolution))

    return test_value, test_null, p_value


def test_statistics(X, Y, model='MMD', kernel_function='rbf', **kwargs):
    """
     This function performs a test statistics and return a test value. This implementation can perform
     the Kolmogorov-Smirnov test (for one-dimensional data only), Kullback-Leibler divergence and MMD.

    :param X: numpy-array
        Data, of size MxD [M is the number of data points, D is the features dimension]
    :param Y: numpy-array
        Data, of size NxD [N is the number of data points, D is the features dimension]
    :param model: string
        defines the basis model to perform two sample test ['KS', 'KL', 'MMD']
    :param kernel_function: string
        defines the kernel function, only used for the MMD.
        For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
    :param kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters or `KL_divergence_estimator()`
        as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    :return: float
        the test value
    """

    if model not in ['KS', 'KL', 'MMD']:
        raise ValueError("The Model '%s' is not implemented, try 'KS', 'KL', or 'MMD'." % model)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices. X and Y should have the same feature dimension,"
                         ": X.shape[1] == %i while Y.shape[1] == %i." % (X.shape[1], Y.shape[1]))


    m = len(X)
    n = len(Y)

    # compute the test statistics according to the input model
    if model == 'MMD':
        XY = np.vstack([X, Y])
        K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
        test_value = MMD2u_estimator(K, m, n)

    return test_value


def witness_function(X, Y, grid, kernel_function='rbf', **kwargs):
    """
     This function computes the witness function. For the definition of the witness function see page 729
     in the "A Kernel Two-Sample Test" by Gretton et al. (2012)

    :param X: numpy-array
        Data, of size MxD [M is the number of data points, D is the features dimension]
    :param Y: numpy-array
        Data, of size NxD [N is the number of data points, D is the features dimension]
    :param gird: numpy-array
        Defines a grid for which the witness function is computed. It has the size PxD
        where P is the number of grid points, D is the features dimension
    :param kernel_function: string
        defines the kernel function, only used for the MMD.
        For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
    :param kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters.
        E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    :return: numpy-array
        witness function
    """

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices. X and Y should have the same feature dimension,"
                         ": X.shape[1] == %i while Y.shape[1] == %i." % (X.shape[1], Y.shape[1]))

    if X.shape[1] != grid.shape[1]:
        raise ValueError("Incompatible dimension for data and grid matrices. data and grid should have the same feature dimension,"
                         ": data.shape[1] == %i while grid.shape[1] == %i." % (X.shape[1], grid.shape[1]))

    # data and grid size
    m = len(X)
    n = len(Y)

    # compute pairwise kernels
    K_xg = pairwise_kernels(X, grid, metric=kernel_function, **kwargs)
    K_yg = pairwise_kernels(Y, grid, metric=kernel_function, **kwargs)

    return (np.sum(K_xg, axis=0) / m) - (np.sum(K_yg, axis=0) / n)


def run():


    # set the random seed
    np.random.seed(100)

    # set the number of observed data points
    n = 200

    # draw random points from a log-normal and a gaussian distributions
    Y = np.random.lognormal(mean=2, sigma=0.3, size=n)
    X = np.random.normal(loc=np.exp(2.0+0.3**2/2.0), scale=0.3*np.exp(2.0), size=n)

    XX = X[:, np.newaxis]
    YY = Y[:, np.newaxis]

    # compute MMD/K-S two sample tests and their null distributions
    sigma2 = np.median(pairwise_distances(XX, YY, metric='euclidean'))**2 * 2.0
    mmd2u, mmd2u_null, p_value = two_sample_test(XX, YY, model='MMD',
                                                 kernel_function='rbf', gamma=1.0/sigma2,
                                                 iterations=5000, verbose=True, n_jobs=1)

    # visualize the results
    plt.figure(figsize=(6, 9))
    plt.subplot(211)
    prob, bins, patches = plt.hist(mmd2u_null, range=[-0.005, 0.015], bins=50,
                                   density=True, color='green', label='Null distribution')
    plt.plot(mmd2u, prob.max()/25, 'wv', markersize=14, markeredgecolor='k',
             markeredgewidth=2, label="MMD2u   p-value = %0.3f"%p_value)
    plt.xlim([-0.005, 0.020])
    plt.ylabel('PDF', size=20)
    plt.legend(loc=1, numpoints=1, prop={'size':15})

    plt.show()
    
    
####################
run()
