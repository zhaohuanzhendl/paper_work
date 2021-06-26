import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def target_function(x):
    return 20 * x - 16 * x ** 2


def gene_dataset():
    dataset_size = 50
    np.random.seed(2021)
    data_x = np.random.uniform(0, 1, dataset_size)
    data_epsilon = np.random.normal(0, 1, dataset_size)
    data_y = target_function(data_x) + data_epsilon * 2 / 5
    return data_x, data_y


class KernelRidge:
    def __init__(self, x, y, lam, kernel_function):
        self.x, self.kernel_function = x, kernel_function

        y_t = np.mat(y)
        dataset_size = len(x)
        lam_i = lam * np.mat(np.identity(dataset_size))
        k = np.mat(np.zeros((dataset_size, dataset_size)))
        for i in range(0, dataset_size):
            for j in range(0, dataset_size):
                k[i, j] = kernel_function(x[i], x[j])

        self.coefficient_vec = 2 * y_t * np.linalg.inv(lam_i + 2 * k)

    def predict(self, x):
        prediction_y = np.zeros(len(x))
        for i in range(0, len(x)):
            kappa = np.mat(np.zeros(len(self.x))).T
            for j in range(0, len(self.x)):
                kappa[j, 0] = self.kernel_function(self.x[j], x[i])
            prediction_y[i] = self.coefficient_vec * kappa
        return prediction_y


def kernel_rbf(x, y):
    return math.exp(-(x - y) ** 2)


def main():
    x, y = gene_dataset()
    #import pdb;pdb.set_trace()
    kernel_ridge = KernelRidge(x, y, 0.01, kernel_rbf)


    x_prediction = np.linspace(0, 1, 100)
    y_prediction = kernel_ridge.predict(x_prediction)

    plt.scatter(x, y)
    plt.plot(x_prediction, y_prediction, 'g')
    plt.plot(x_prediction, target_function(x_prediction), 'r')
    plt.show()


if __name__ == '__main__':
    main()
