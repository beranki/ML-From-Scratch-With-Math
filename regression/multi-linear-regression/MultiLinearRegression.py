import numpy as np

class MultiLinearRegression:
    def __init__(self, alpha, epochs, W = None, b = 0):  
        self.alpha = alpha
        self.epochs = epochs
        self.W = W
        self.b = b

    def fit(self, X, y):
        n = X.shape[0] #number of samples
        m = len(y)
        self.W = np.zeros(X.shape[1]).reshape(-1, 1) #initializing weights to same size as X

        for epoch in range(self.epochs): #repeat gradient descent till convergence
            y_pred = np.dot(X, self.W) + self.b
            dW = 1/m * np.dot(X.T, (y_pred - y)) #computing the gradient of W
            db = 1/m * np.sum(y_pred - y)

            self.W -= self.alpha * dW #adjusting the weights
            self.b -= self.alpha * db #adjusting the biases

        return self.W, self.b

    def predict(self, X):
        return np.dot(X, self.W) + self.b
