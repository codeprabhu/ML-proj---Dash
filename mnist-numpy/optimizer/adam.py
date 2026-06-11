import numpy as np
from .optmizer import Optimizer
class Adam(Optimizer):

    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = {}
        self.v = {}
        L = len(parameters)// 2
        for l in range(1, L+1):
            self.m[f"W{l}"] = np.zeros_like(parameters[f"W{l}"])
            self.m[f"b{l}"] = np.zeros_like(parameters[f"b{l}"])
            self.v[f"W{l}"] = np.zeros_like(parameters[f"W{l}"])
            self.v[f"b{l}"] = np.zeros_like(parameters[f"b{l}"])
        self.t = 0
        
    def step(self, parameters, grads):
        self.t += 1
        L = len(parameters)//2
        for l in range(1, L+1):
            self.m[f"W{l}"] = self.m[f"W{l}"]*self.beta1 + (1-self.beta1)*grads[f"dW{l}"]
            self.m[f"b{l}"] = self.m[f"b{l}"]*self.beta1 + (1-self.beta1)*grads[f"db{l}"]

            self.v[f"W{l}"] = self.v[f"W{l}"]*self.beta2 + (1-self.beta2)*(grads[f"dW{l}"]**2)
            self.v[f"b{l}"] = self.v[f"b{l}"]*self.beta2 + (1-self.beta2)*(grads[f"db{l}"]**2)

            mW_hat = self.m[f"W{l}"]/(1-self.beta1**self.t)
            mb_hat = self.m[f"b{l}"]/(1-self.beta1**self.t)
            vW_hat = self.v[f"W{l}"]/(1-self.beta2**self.t)
            vb_hat = self.v[f"b{l}"]/(1-self.beta2**self.t)

            parameters[f"W{l}"] -= mW_hat*self.learning_rate/(np.sqrt(vW_hat)+self.epsilon)
            parameters[f"b{l}"] -= mb_hat*self.learning_rate/(np.sqrt(vb_hat)+self.epsilon)

        return parameters