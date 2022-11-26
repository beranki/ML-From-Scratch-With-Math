import numpy as np

class LogisticRegression() :
    def __init__( self, lr, epochs, threshold=0.5) :        
        self.lr = lr        
        self.epochs = epochs
        self.threshold = threshold
          
    # Function for model training    
    def fit( self, X, y) :        
        self.m, self.n = X.shape        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.y = y
        #self.costs = []

        for i in range( self.epochs) : 
            z = self.X @ self.W + self.b           
            p = 1/(1 + np.exp(-z))      

            #self.costs.append(self.cost(self.y, p))

            dW = np.dot(self.X.T, np.reshape((p - self.y.T), self.m))         
            db = np.sum(np.reshape((p - self.y.T), self.m))
            
            # update weights    
            self.W -= self.lr * dW    
            self.b -= self.lr * db         

        return self.W, self.b
    
    def cost(self, y, p):
        print(np.log(p))
        return -np.sum(y*np.log(p) + (1-y) * np.log(1-p))

    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        return np.where(Z > self.threshold, 1, 0)
  