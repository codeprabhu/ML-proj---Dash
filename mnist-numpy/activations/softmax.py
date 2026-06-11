import numpy as np
from .activations import Activation

class Softmax(Activation):

    def forward(self, Z):

        Z = Z - np.max(
            Z,
            axis=1,
            keepdims=True
        )

        exp_Z = np.exp(Z)

        return (
            exp_Z
            /
            np.sum(
                exp_Z,
                axis=1,
                keepdims=True
            )
        )

    def backward(self, Z, dA):

        Y_hat = self.forward(Z)

        m = Y_hat.shape[0]

        dZ = np.zeros_like(Y_hat)

        for i in range(m):

            s = Y_hat[i]

            jacobian = (
                np.diag(s)
                -
                np.outer(s, s)
            )

            dZ[i] = (
                jacobian
                @
                dA[i]
            )

        return dZ