import numpy as np
def backward(X,y, parameters, cache, hidden_activation, output_activation, loss):
    grads = {}
    m = X.shape[0]
    L = len(parameters)//2
    Y_hat = cache[f"A{L}"]
    dA = loss.backward(y, Y_hat)
    
    dZ = output_activation.backward(cache[f"Z{L}"], dA)
    grads[f"dW{L}"] = (cache[f"A{L-1}"].T @ dZ)/m
    grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True)/m

    for l in reversed(range(1,L)):
        dA = (dZ @ parameters[f"W{l+1}"].T)
        dZ = hidden_activation.backward(cache[f"Z{l}"], dA)
        grads[f"dW{l}"] = (cache[f"A{l-1}"].T @ dZ)/m
        grads[f"db{l}"] = (np.sum(dZ, axis = 0, keepdims=True))/m

    return grads
