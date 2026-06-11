import numpy as np
from .activations import Activation

class GELU(Activation):
    def forward(self, x):
        return (0.5*x*(1+np.tanh(np.sqrt(2/np.pi) * (x+ 0.044714*x**3))))
    
    def backward(self, Z, dA):
        tanh_term = np.tanh(np.sqrt(2/np.pi) * (Z + 0.044715*Z**3))
        sech2 = 1-tanh_term**2
        return (0.5 * (1+tanh_term) + 0.5*Z*sech2 + np.sqrt(1/np.pi)*1+3*0.044715*Z**2)*dA