import numpy as np

class SVM:
    def __init__(self, C=1.0):
        self.C = C #C is the # of errors in training
        self.W = 0
        self.b = 0

    def hinge_loss(self, X, y):
        regularizer = self.W*self.W/2
        for i in range(X.shape[0]):
            loss = regularizer + self.C*max(0, 1-y[i]*(np.dot(self.W, X[i])+self.b)) #adds regularizer metric to error term
        return loss[0][0]

    def fit(self, X, y, batch_size, lr, epochs):
        n = X.shape[0]
        m = X.shape[1]

        ids = np.arange(n)
        np.random.shuffle(ids)

        self.W = np.zeros((1,m))
        self.b = 0
        self.losses = []

        for i in range(epochs):
            l = self.hinge_loss(X, y)
            self.losses.append(l)

            for j in range(0, n, batch_size): #iterating from 0 -> # of samples every batch size
                dW = 0
                db = 0

                for a in range(j, j + batch_size): #iterating through every elmt in the batch
                    if (a < n):
                        if (y[ids[a]] * np.dot(self.W, X[ids[a]].T) + self.b) <= 1:
                            dW += self.C * y[ids[a]] * X[ids[a]]
                            db += self.C * y[ids[a]]

                    self.W -= lr*(self.W - dW)
                    self.b += lr*db

        return self.W, self.b, self.losses


    def predict(self, X):
        return np.sign(np.dot(X, self.W[0]) + self.b)