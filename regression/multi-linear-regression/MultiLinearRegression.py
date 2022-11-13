import numpy as np

class MultiLinearRegression:
    def __init__(self, alpha, epochs, W = None, b = 0):  
        self.alpha = alpha
        self.epochs = epochs
        self.W = W
        self.b = b

    def fit(self, X, y):
        costs = [0]*self.epochs
        self.m = len(y) #number of features 
        self.W = np.zeros(X.shape[1]).reshape(-1, 1) #initializing weights to same size as X

        for epoch in range(self.epochs): #repeat gradient descent for # of epochs (also opt for reaching convergence)
            y_pred = np.dot(X, self.W) + self.b
            dW = 2/self.m * np.dot(X.T, (y_pred - y)) #computing the gradient of W
            db = 2/self.m * np.sum(y_pred - y) #computing the gradient of b

            self.W -= self.alpha * dW #adjusting the weights
            self.b -= self.alpha * db #adjusting the biases

            costs[epoch] = self.compute_cost(X, y)

        return self.W, self.b, costs

    def compute_cost(self, X, y):
        predictions = X.dot(self.W)
        errors = np.subtract(predictions, y)
        sqrErrors = np.square(errors) 
        J = 1 / (2 * self.m) * np.sum(sqrErrors)

        return J

    def predict(self, X):
        return np.dot(X, self.W) + self.b
