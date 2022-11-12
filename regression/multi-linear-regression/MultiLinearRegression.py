class MultiLinearRegression:
    def __init__(self, lr, epochs, W = None, b = 0):  
        self.lr = lr
        self.epochs = epochs
        self.W = W
        self.b = b