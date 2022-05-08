#!/usr/bin/env python
# coding: utf-8

# A notebook to test and demonstrate the `MMD test` of Gretton et al., 2012 used as a goodness-of-fit test. Require the ability to sample from the density `p`.

# In[106]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')
#%config InlineBackend.figure_format = 'svg'
#%config InlineBackend.figure_format = 'pdf'

import freqopttest.tst as tst
import kgof
import kgof.data as data
import kgof.density as density
import kgof.goftest as gof
import kgof.mmd as mgof
import kgof.kernel as ker
import kgof.util as util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# In[107]:


seed = 13
d = 3
alpha = 0.05

mean = np.zeros(d)
variance = 1


#---------------------------------------------------------------
#原始实现 begin
#---------------------------------------------------------------
import autograd.numpy as np
from past.utils import old_div

#fortest code block
x1 = np.array([[1,2,3],[2,3,3]])
x2 = np.array([[1,1,1], [2,2,2]])
print(np.sum(x1**2, 1)[:, np.newaxis])
print(-2*np.dot(x1,x2.T))
print(np.sum(x2**2,1)[:, np.newaxis])
print(np.sum(x1**2, 1)[:, np.newaxis]-2*np.dot(x1,x2.T)+np.sum(x2**2,1)[:, np.newaxis])
#np.diag()

# In[141]:


class KGauss():

    def __init__(self, sigma2=1):
        assert sigma2 > 0, 'sigma2 must be positive >0 !'
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        当X1==X2时, 功能相当于: scipy.spatial.distance.pdist(X1, 'sqeuclidean')
        ----------------------------
        eval gauss kernel on 2d numpy arrays.
        INPUT:
        ----------------------------
        X1: (n1, d)shape numpy array
        X2: (n2, d)shape numpy array
        ----------------------------
        RETURN:
        K : (n1, n2)Gram matrix
        """
        (n1,d1) = X1.shape
        (n2,d2) = X2.shape
        assert d1==d2, 'd of two inputs must be the same'
        D2 = np.sum(X1**2, 1)[:, np.newaxis] -2*np.dot(X1, X2.T) + np.sum(X2**2, 1)
        K = np.exp(old_div(-D2, self.sigma2))
        return K


    def pair_eval(self, X, Y):
        """
        input:  X, Y : n x d numpy array
        Evaluate k(x1, y1), k(x2, y2), ...

        Return : a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(old_div(-D2,self.sigma2))
        return Kvec

#k = KGauss(sigma2=1)
#print(k.pair_eval(x1,x2))
#kxy = k.eval(x1, x2)
#print(kxy)
#print(np.sum(kxy))
#print(np.diag(kxy))


class MMDTest():
    """
    Quadratic MMD test where the null distribution is computed by permutation.
    Use a single U-statistic i.e., remove diagonal from the Kxy matrix.
    ref: https://github.com/wittawatj/interpretable-test
    ------------
    simplify the compute process of mmd, only get mmd^2
    """
    def __init__(self, kernel, n_permute=400, alpha=0.01):
        self.kernel = kernel
        self.n_permute = n_permute
        self.alpha = alpha

    def perform_test(self, X,Y):
        """compute mmd2, pvalue and reject status...
        """
        (_, d) = X.shape
        alpha = self.alpha
        mmd2_stat = self.compute_stat(X, Y)
        print(mmd2_stat)
        #k = self.kernel
        #repeats = self.n_permute
        repeats = 4
        list_mmd2 = self._permutation_list_mmd2(X, Y, repeats)
        print(list_mmd2)
        print(np.mean(list_mmd2 > mmd2_stat))
        pvalue = np.mean(list_mmd2 > mmd2_stat) #统计list_mmd2中大于mmd2_stat的个数占比
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': mmd2_stat,  'h0_rejected': pvalue < alpha}
        return results

    def compute_stat(self, X, Y):
        """
        Compute the test statistic: empirical quadratic MMD^2,  not include p-value
        """
        assert X.shape[0]==Y.shape[0], 'nx must be the same ny'
        k = self.kernel
        Kx = k.eval(X, X)
        Ky = k.eval(Y, Y)
        Kxy = k.eval(X, Y)
        mmd2 = self._h1_mean(Kx, Ky, Kxy)
        return mmd2


    def _h1_mean(self, Kx, Ky, Kxy):
        nx = Kx.shape[0]
        ny = Ky.shape[0]
        xx = old_div((np.sum(Kx) - np.sum(np.diag(Kx))),(nx*(nx-1)))
        yy = old_div((np.sum(Ky) - np.sum(np.diag(Ky))),(ny*(ny-1)))
        xy = old_div((np.sum(Kxy) - np.sum(np.diag(Kxy))),(nx*(ny-1)))
        #import pdb; pdb.set_trace()
        mmd2 = xx - 2*xy + yy # paper Lemma 6, equation 3
        return mmd2

    def _permutation_list_mmd2(self, X, Y, repeats, seed=8273):

        k = self.kernel
        XY = np.vstack((X,Y))
        Kxyxy = k.eval(XY, XY)
        rand_state = np.random.get_state()
        np.random.seed(seed)

        nxy = XY.shape[0]
        nx = X.shape[0]
        ny = Y.shape[0]
        list_mmd2 = np.zeros(repeats)

        for r in range(repeats):
            ind = np.random.choice(nxy, nxy, replace=False) #可以理解为进行了一次随机排列
            # divide into new X, Y
            indx = ind[:nx]
            indy = ind[nx:]
            Kx = Kxyxy[np.ix_(indx, indx)] #获取 遍历indx indy 中元素两两相交后组成的位置(x,y) 去Kxyxy中捞取该位置中的元素
            Ky = Kxyxy[np.ix_(indy, indy)]
            Kxy = Kxyxy[np.ix_(indx, indy)]
            mmd2r = self._h1_mean(Kx, Ky, Kxy)
            list_mmd2[r] = mmd2r

        np.random.set_state(rand_state)  #????
        return list_mmd2

#FOR TEST
k = KGauss(1)
mmd_test = MMDTest(k)
mmd_test.perform_test(x1, x2)

#end MMD
#--------------------------------------------


#----------------------------------------------------
# ## Test original implementation
# Original implementation of Chwialkowski et al., 2016

from scipy.spatial.distance import squareform, pdist

def simulatepm(N, p_change):
    '''

    :param N:
    :param p_change:
    :return:
    '''
    X = np.zeros(N) - 1
    change_sign = np.random.rand(N) < p_change
    for i in range(N):
        if change_sign[i]:
            X[i] = -X[i - 1]
        else:
            X[i] = X[i - 1]
    return X


class _GoodnessOfFitTest:
    def __init__(self, grad_log_prob, scaling=1):
        #scaling is the sigma^2 as in exp(-|x_y|^2/2*sigma^2)
        self.scaling = scaling*2
        self.grad = grad_log_prob
        # construct (slow) multiple gradient handle if efficient one is not given


    def grad_multiple(self, X):
        #print self.grad
        return np.array([(self.grad)(x) for x in X])

    def kernel_matrix(self, X):

        # check for stupid mistake
        assert X.shape[0] > X.shape[1]

        sq_dists = squareform(pdist(X, 'sqeuclidean'))

        K = np.exp(-sq_dists/ self.scaling)
        return K

    def gradient_k_wrt_x(self, X, K, dim):

        X_dim = X[:, dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1, len(X_dim))

        return -2.0 / self.scaling * K * differences

    def gradient_k_wrt_y(self, X, K, dim):
        return -self.gradient_k_wrt_x(X, K, dim)

    def second_derivative_k(self, X, K, dim):
        X_dim = X[:, dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1, len(X_dim))

        sq_differences = differences ** 2

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

    def get_statistic_multiple_dim(self, samples, dim):
        num_samples = len(samples)

        log_pdf_gradients = self.grad_multiple(samples)
        # n x 1
        log_pdf_gradients = log_pdf_gradients[:, dim]
        # n x n
        K = self.kernel_matrix(samples)
        assert K.shape[0]==K.shape[1]
        # n x n
        gradient_k_x = self.gradient_k_wrt_x(samples, K, dim)
        assert gradient_k_x.shape[0] == gradient_k_x.shape[1]
        # n x n
        gradient_k_y = self.gradient_k_wrt_y(samples, K, dim)
        # n x n
        second_derivative = self.second_derivative_k(samples, K, dim)
        assert second_derivative.shape[0] == second_derivative.shape[1]

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(num_samples, 1) \
                                 * log_pdf_gradients.reshape(1, num_samples)
        A = pairwise_log_gradients * K

        B = gradient_k_x * log_pdf_gradients
        C = (gradient_k_y.T * log_pdf_gradients).T
        D = second_derivative

        V_statistic = A + B + C + D
        #V_statistic =  C

        stat = num_samples * np.mean(V_statistic)
        return V_statistic, stat

    def compute_pvalues_for_processes(self, U_matrix, chane_prob, num_bootstrapped_stats=300):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        with util.NumpySeedContext(seed=10):
            for proc in range(num_bootstrapped_stats):
                # W = np.sign(orsetinW[:,proc])
                W = simulatepm(N, chane_prob)
                WW = np.outer(W, W)
                st = np.mean(U_matrix * WW)
                bootsraped_stats[proc] = N * st

        stat = N * np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats

    def is_from_null(self, alpha, samples, chane_prob):
        dims = samples.shape[1]
        boots = 10 * int(dims / alpha)
        num_samples = samples.shape[0]
        U = np.zeros((num_samples, num_samples))
        for dim in range(dims):
            U2, _ = self.get_statistic_multiple_dim(samples, dim)
            U += U2

        p = self.compute_pvalues_for_processes(U, chane_prob, boots)
        return p, U




sigma = np.array([[1, 0.2, 0.1], [0.2, 1, 0.4], [0.1, 0.4, 1]])

def grad_log_correleted(x):
    #sigmaInv = np.linalg.inv(sigma)
    #return - np.dot(sigmaInv.T + sigmaInv, x) / 2.0
    return -(x-mean)/variance

qm = _GoodnessOfFitTest(grad_log_correleted, scaling=sig2)
X = np.random.multivariate_normal([0, 0, 0], sigma, 200)
print(X.shape)
p_val, U = qm.is_from_null(0.05, X, 0.1)
print("p_val:", p_val)

