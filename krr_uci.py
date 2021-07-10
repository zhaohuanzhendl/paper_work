import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform

def gene_dataset():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header=None)
    #df = pd.read_csv('abalone.data', header=None)
    df.iloc[:, 0] = 1 #忽略第一列类别分类
    xmat = np.mat(df.iloc[:, :-1].values)
    ymat = np.mat(df.iloc[:, -1].values).T
    print(xmat.shape, ymat.shape)
    return xmat, ymat


def RidgeRegression(feature, label, lam):
    '''
    input:  feature(mat):
            label(mat):
    output: w(mat):
    '''
    n = np.shape(feature)[1]
    w = (feature.T * feature + lam * np.mat(np.eye(n))).I * feature.T * label
    return w

def kernel_rbf(x, gamma=0.1):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma * mat_sq_dists)


def KernelRidge(x, y, kernel_function, lam=0.1):

    #print(x)
    K = kernel_function(x)
    N = K.shape[0]
    In = np.eye(N)
    #lam = 0.2
    alpha = np.linalg.inv(K + lam * In).dot(y)
    w = alpha.T.dot(x)
    #print(w)
    return w


def sseCal(x, y, w):
    n = x.shape[0]
    #w = regres(dataSet)
    #print(n)
    y_p = x * w.T  #全是mat 不是np.array
    print(y[:, :10])
    print(y_p[:, :10])
    y_p = y_p.reshape([n,])
    rss = np.power(y_p - y, 2).sum()
    return rss



def main():
    x, y = gene_dataset()
    #for lam in [0.001, 0.01, 0.1, 1, 10,20,50,100, 1000, 10000, 100000]:
    for lam in [100, 1000, 10000, 100000]:
        w = KernelRidge(x, y, kernel_rbf, lam)
        rss = sseCal(x,y, w)
        print(lam, rss)

    #plt.scatter(x, y)
    #plt.plot(x_prediction, y_prediction, 'g')
    #plt.plot(x_prediction, target_function(x_prediction), 'r')
    #plt.show()

if __name__ == '__main__':
    main()
