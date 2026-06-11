from activations import *

def forward(x,parameters, hidden_activation, output_activation):
    cache = {}
    A = x
    cache["A0"] = x
    L = len(parameters)//2

    for l in range(1, L):
        Z = (A@parameters[f"W{l}"] + parameters[f"b{l}"])
        A = hidden_activation.forward(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    
    ZL = (A@parameters[f"W{L}"] + parameters[f"b{L}"])
    Y_hat= output_activation.forward(ZL)

    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = Y_hat

    return Y_hat, cache