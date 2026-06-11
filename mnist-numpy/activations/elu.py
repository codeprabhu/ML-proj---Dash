import numpy as np
from .activations import Activation

class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self,x):
        return np.where(x>0, x, self.alpha*(np.exp(x)-1))
    
    def backward(self, Z, dA):
        return np.where(Z > 0, 1, self.alpha*np.exp(Z))*dA