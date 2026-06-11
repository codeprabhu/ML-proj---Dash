import numpy as np
from .activations import Activation

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self,x):
        return np.where(x>0, x, self.alpha*x)
    
    def backward(self,Z, dA):
        return np.where(Z>0, 1.0, self.alpha)*dA