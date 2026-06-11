import numpy as np
from .loss import Loss 

class MAE(Loss):
    def forward(self, y_true, y_pred):
        return np.mean(np.abs(y_true-y_pred))
    
    def backward(self, y_true, y_pred):
        return np.sign(y_pred-y_true)/y_true.size