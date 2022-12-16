import numpy as np

class SVM:
    def __init__(self, C=1.0):
        self.C = C #C is the error term
        self.W = 0
        self.b = 0

    
