import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from   sklearn.gaussian_process.kernels import RBF
from   sklearn.kernel_ridge import KernelRidge

from   sklearn.exceptions import NotFittedError
from   sklearn.linear_model import Ridge

class RFFRidgeRegression:

    def __init__(self, rff_dim=1, alpha=1.0, sigma=1.0):
        """Kernel ridge regression using random Fourier features.
        rff_dim : Dimension of random feature.
        alpha :   Regularization strength. Should be a positive float.
        """
        self.fitted  = False
        self.rff_dim = rff_dim
        self.sigma   = sigma
        self.lm      = Ridge(alpha=alpha)
        self.b_      = None
        self.W_      = None

    def fit(self, X, y):
        """Fit model with training data X and target y.
        """
        Z, W, b = self._get_rffs(X, return_vars=True)
        self.lm.fit(Z.T, y)
        self.b_ = b
        self.W_ = W
        self.fitted = True
        return self

    def predict(self, X):
        """Predict using fitted model and testing data X.
        """
        if not self.fitted:
            msg = "Call 'fit' with appropriate arguments first."
            raise NotFittedError(msg)
        Z = self._get_rffs(X, return_vars=False)
        return self.lm.predict(Z.T)

    def _get_rffs(self, X, return_vars):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        if self.W_ is not None:
            W, b = self.W_, self.b_
        else:
            W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
            b = np.random.uniform(0, 2*np.pi, size=self.rff_dim)

        B    = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1./ np.sqrt(self.rff_dim)
        Z    = norm * np.sqrt(2) * np.cos(self.sigma * W @ X.T + B)

        if return_vars:
            return Z, W, b
        return Z

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
cmap      = plt.cm.get_cmap('Blues')
#print(X,y)
ax1.scatter(X, y, s=30, c=[cmap(0.3)])
ax2.scatter(X, y, s=30, c=[cmap(0.3)])

# Fit kernel ridege regression using an RBF kernel.
clf    = KernelRidge(kernel=RBF())
clf    = clf.fit(X, y)
y_pred = clf.predict(X_test)
ax1.plot(X_test, y_pred, c=cmap(0.9))

# Fit kernel ridge regression using random Fourier features.
rff_dim = 20
clf     = RFFRidgeRegression(rff_dim=rff_dim)
clf.fit(X, y)
y_pred  = clf.predict(X_test)
ax2.plot(X_test, y_pred, c=cmap(0.9))

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
