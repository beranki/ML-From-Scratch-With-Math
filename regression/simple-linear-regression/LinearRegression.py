import numpy as np

class LinearRegression:
    def __init__(self, a, epochs):
        self.a = a
        self.epochs = epochs
        self.b_1 = 0
        self.b_0 = 0

    def fit(self, X, y):
        self.n = np.size(X) #number of samples
        SS_xy = np.sum(y*X) - self.n*np.mean(X)*np.mean(y) #calculating SSxy (numerator of B1)
        SS_xx = np.sum(X*X) - self.n*np.mean(X)*np.mean(X) #calculating SSxx (denominator of B1)

        self.b_1 = SS_xy / SS_xx #calculates B1
        self.b_0 = np.mean(y) - self.b_1*np.mean(X) #uses B1 to calculate B0 

        return self.b_0, self.b_1 

    def predict(self, X):
        return self.b_1 * X + self.b_0
