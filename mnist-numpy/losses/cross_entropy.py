import numpy as np
from .loss import Loss

class CrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        return (-np.sum(y_true*np.log(y_pred))/y_true.shape[0])
    
    def backward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        return -y_true/y_pred