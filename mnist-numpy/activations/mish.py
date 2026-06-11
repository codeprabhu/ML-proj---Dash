import numpy as np
from .activations import Activation

class Mish(Activation):
    def softplus(self,x):
        return np.log1p(np.exp(x))
    
    def forward(self, x):
        return x*np.tanh(self.softplus(x))
    
    def backward(self, Z, dA):
        sp = self.softplus(Z)
        tanh_sp = np.tanh(sp)

        sigmoid = (1/1+np.exp(-Z))
        return (tanh_sp + Z*sigmoid*(1-tanh_sp**2))*dA