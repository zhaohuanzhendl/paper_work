import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from  sklearn.gaussian_process.kernels import RBF
from  sklearn.kernel_ridge import KernelRidge

from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.linear_model import Ridge

class RFMRidgeRegression:

    def __init__(self, n_components=50, p=2, distribution='rademacher', gamma='auto', coefs=None,max_expansion=20, random_state=None):
        """
        n_components : int (default=50)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

        p : int (default=10)
        Parameter of the distribution that determines which components of the
        Maclaurin series are approximated.

        distribution : str, (default="rademacher")
        Distribution for random_weights_.
        "rademacher", "gaussian", "laplace", "uniform", or "sparse_rademacher"
        can be used.

        gamma : float or str (default="auto")
        Parameter of the exponential kernel.

        coefs : list-like (default=None)
        list of coefficients of Maclaurin expansion.

        max_expansion : int (default=20)
        Threshold of Maclaurin expansion.
        
        Attributes
        ----------
        orders_ : array, shape (n_components, )
        The sampled orders of the Maclaurin expansion.
        The j-th components of random feature approximates orders_[j]-th order
        of the Maclaurin expansion.

        random_weights_ : array, shape (n_features, np.sum(orders_))
        The sampled basis.
        ----------
        """
        self.lm = Ridge(alpha=1)
        self.n_components = n_components
        self.p = p
        self.gamma = gamma
        self.distribution = distribution
        # coefs of Maclaurin series.
        # If kernel is 'poly' or 'exp', this is computed automatically.
        self.coefs = coefs
        self.max_expansion = max_expansion
        self.p_choice = None
        #self.h01 = h01
        #self.dense_output = dense_output
        self.random_state = random_state

    def _rademacher(self, random_state, size):
        return random_state.randint(2, size=size, dtype=np.int32)*2-1    
    
    def get_random_matrix(self, random_state, distribution, size, dtype=np.float64):
        if distribution == 'rademacher':
            return self._rademacher(random_state, size).astype(dtype)
        
    def _sample_orders(self, random_state):
        coefs = np.array(self.coefs)
        if self.p_choice is None:
            p_choice = (1/self.p) ** (np.arange(len(coefs)) + 1)
            if np.sum(coefs == 0.) != 0:
                p_choice[coefs == 0] = 0
            p_choice /= np.sum(p_choice)
            self.p_choice = p_choice

        self.orders_ = random_state.choice(len(self.p_choice),
                                           self.n_components,
                                           p=self.p_choice).astype(np.int32)    
    
    def fit(self, X, y=None):
        """Fit model with training data X and target y.
        """
        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
            
        if self.coefs is None:
            self.coefs = gamma ** np.arange(self.max_expansion)
            self.coefs /= factorial(range(self.max_expansion))
            
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        
        self._sample_orders(random_state)
        distribution = self.distribution.lower()
        size = (n_features, np.sum(self.orders_))
        self.random_weights_ = self.get_random_matrix(random_state, distribution, size)
        
        return self
        

    def predict(self, X):
        """Predict using fitted model and testing data X.
        """

        Z = self.transform(X, return_vars=False)
        return self.lm.predict(Z.T)

 

N     = 100
X     = np.linspace(-10, 10, N)[:, None]
mean  = np.zeros(N)
#print(X)
cov   = RBF()(X.reshape(N, -1))
#print(cov.shape, cov)
#print(X.reshape(N, -1))
y     = np.random.multivariate_normal(mean, cov)
noise = np.random.normal(0, 0.5, N)
y    += noise
print(cov.shape)
# Finer resolution for smoother curve visualization.
X_test = np.linspace(-10, 10, N*2)[:, None]
#print(X_test)

# Set up figure and plot data.
fig, axes = plt.subplots(2, 1)
fig.set_size_inches(10, 5)
ax1, ax2  = axes
#print(X,y)

ax1.scatter(X, y, s=30, c='b')
ax2.scatter(X, y, s=30, c='b')

# Fit kernel ridege regression using an RBF kernel.
clf    = KernelRidge(kernel=RBF())
clf    = clf.fit(X, y)
y_pred = clf.predict(X_test)
ax1.plot(X_test, y_pred, c='r')

# Fit kernel ridge regression using random Fourier features.
rff_dim = 20
clf     = RFMRidgeRegression()
clf.fit(X, y)
y_pred  = clf.predict(X_test)
ax2.plot(X_test, y_pred, c='r')

# Labels, etc.
ax1.margins(0, 0.1)
ax1.set_title('RBF kernel regression')
ax1.set_ylabel(r'$y$', fontsize=14)
ax1.set_xticks([])
ax2.margins(0, 0.1)
ax2.set_title(rf'RFF ridge regression, $R = {rff_dim}$')
ax2.set_ylabel(r'$y$', fontsize=14)
ax2.set_xlabel(r'$x$', fontsize=14)
ax2.set_xticks(np.arange(-10, 10.1, 1))
plt.tight_layout()
plt.show()

import numpy as np
mean = (1, 2)
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, (3, 3))
print(x)
xx = np.random.multivariate_normal(mean, cov)
print(xx)
