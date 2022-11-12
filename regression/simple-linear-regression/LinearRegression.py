import numpy as np

class LinearRegression:
    def __init__(self, a, epochs, W=None, b=0):
        self.a = a
        self.epochs = epochs
        self.W = W
        self.b = b

    def fit(self, X, y):
        self.n = np.size(X)
        SS_xy = np.sum(y*X) - self.n*np.mean(X)*np.mean(y)
        SS_xx = np.sum(X*X) - self.n*np.mean(X)*np.mean(X)

        b_1 = SS_xy / SS_xx
        b_0 = np.mean(y) - b_1*np.mean(X)

        return b_0, b_1

    def predict(self, X):
        return self.W * X + self.b
