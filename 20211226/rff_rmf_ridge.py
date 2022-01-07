import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
#from   rffridge import RFFRidgeRegression
from   sklearn.gaussian_process.kernels import RBF
from   sklearn.kernel_ridge import KernelRidge

from   sklearn.exceptions import NotFittedError
from   sklearn.linear_model import Ridge

from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from scipy.special import factorial, binom

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
        print("Z.shape:", Z.shape)
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

class RandomMaclaurin():

    def __init__(self, D=50, p=2, kernel='exp',
                 gamma=1.0, coefs=None, 
                 max_expansion=50,
                 random_state=123, 
                 alpha=1.0):

        self.D = D
        self.p = p
        self.gamma = gamma
        # coefs of Maclaurin series.
        self.coefs = coefs
        self.kernel = kernel
        self.max_expansion = max_expansion
        self.p_choice = None
        self.random_state = random_state
        self.fitted = False
        self.lm = Ridge(alpha=alpha)

    def _set_coefs(self, gamma):
        if self.coefs is None:
            if self.kernel == 'exp':
                self.coefs = gamma ** np.arange(self.max_expansion)
                self.coefs /= factorial(range(self.max_expansion))
            else:
                raise ValueError("When using the user-specific kernel "
                                 "function, coefs must be given explicitly.")

    def _sample_orders(self, random_state):
        coefs = np.array(self.coefs)

        if self.p_choice is None:
            p_choice = (1/self.p) ** (np.arange(len(coefs)) + 1)
            if np.sum(coefs == 0.) != 0:
                p_choice[coefs == 0] = 0
            p_choice /= np.sum(p_choice)
            self.p_choice = p_choice

        self.orders_ = random_state.choice(len(self.p_choice),
                                           self.D,
                                           p=self.p_choice).astype(np.int32) + 1

    def _rademacher(self, random_state, size):
        return random_state.randint(2, size=size, dtype=np.int32)*2-1


    def fit(self, X, y=None):

        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=False)
        n_samples, n_features = X.shape

        if self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma

        self._set_coefs(gamma)
        self._sample_orders(random_state)
        #distribution = self.distribution.lower()
        size = (n_features, np.sum(self.orders_))
        self.random_weights_ = self._rademacher(random_state,size).astype(np.float64)
        self.fitted = True
        
        Z = self.transform(X)
        self.lm.fit(Z.T, y)
        return self


    def transform_implement(self, X):

        n_samples, n_features = X.shape
        #Z = np.zeros((n_samples, self.D))
        Z = np.ones((n_samples, self.D))
        print(self.orders_)
        #print(self.random_weights_)
        #print(self.random_weights_.shape)
        for row in range(n_samples):
            sample = X[row]
            weight_offset = 0
            for i in range(self.D):
                #weight_offset = 0
                N = self.orders_[i]
                a_N = self.coefs[N]
                for j in range(1, N+1):
                    #print("j:", j)
                    #print("N:", N)
                    #print("weight_offset:", weight_offset)
                    Z[row][i] *= np.sum(self.random_weights_[:, weight_offset] * sample)
                    weight_offset += 1
                Z[row][i] *= np.sqrt(a_N * (self.p**(N+1)))

            #print("weight_offset:", weight_offset)
        print("rmf Z.T.shape:", Z.T.shape)
        #print("Z:", Z)
        return Z.T

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, D)
        """
        #check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        X_new = self.transform_implement(X)

        return X_new
    
    def predict(self, X):
        """Predict using fitted model and testing data X.
        """
        if not self.fitted:
            msg = "Call 'fit' with appropriate arguments first."
            raise NotFittedError(msg)
        Z = self.transform(X)
        return self.lm.predict(Z.T)

N     = 100
X     = np.linspace(-10, 10, N)[:, None]
mean  = np.zeros(N)
print(X.shape, X.reshape(N, -1).shape)
cov   = RBF()(X.reshape(N, -1))
print(cov.shape)
#print(X.reshape(N, -1))
y     = np.random.multivariate_normal(mean, cov)
noise = np.random.normal(0, 0.5, N)
y    += noise
# Finer resolution for smoother curve visualization.
X_test = np.linspace(-10, 10, N*2)[:, None]
#print(X_test)

# Set up figure and plot data.
fig, axes = plt.subplots(3, 1)
fig.set_size_inches(10, 10)
ax1, ax2, ax3  = axes
cmap      = plt.cm.get_cmap('Blues')
#print(X,y)
ax1.scatter(X, y, s=30, c=[cmap(0.3)])
ax2.scatter(X, y, s=30, c=[cmap(0.3)])
ax3.scatter(X, y, s=30, c=[cmap(0.3)])

# Fit kernel ridege regression using an RBF kernel.
clf_1    = KernelRidge(kernel=RBF())
clf_1    = clf.fit(X, y)
y_pred = clf_1.predict(X_test)
ax1.plot(X_test, y_pred, c=cmap(0.9))

# Fit kernel ridge regression using random Fourier features.
rff_dim = 20
clf_2 = RFFRidgeRegression(rff_dim=rff_dim)
clf_2.fit(X, y)
y_pred  = clf_2.predict(X_test)
ax2.plot(X_test, y_pred, c=cmap(0.9))

D=100
clf_3 = RandomMaclaurin(D=D)
clf_3.fit(X, y)
y_pred  = clf_3.predict(X_test)
ax3.plot(X_test, y_pred, c=cmap(0.9))

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

ax3.margins(0, 0.1)
ax3.set_title(rf'RMF regression, $D = {50}$')
ax3.set_ylabel(r'$y$', fontsize=14)
ax3.set_xlabel(r'$x$', fontsize=14)
ax3.set_xticks(np.arange(-10, 10.1, 1))

plt.tight_layout()
plt.show()


