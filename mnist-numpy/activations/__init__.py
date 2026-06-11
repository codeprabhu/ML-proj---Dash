from .relu import ReLU
from .leaky_relu import LeakyReLU
from .prelu import PReLU
from .sigmoid import Sigmoid
from .tanh import Tanh
from .elu import ELU
from .swish import Swish
from .gelu import GELU
from .mish import Mish
from .softmax import Softmax

ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "prelu": PReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "elu": ELU,
    "swish": Swish,
    "gelu": GELU,
    "mish": Mish
}