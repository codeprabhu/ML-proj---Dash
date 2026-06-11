from .cross_entropy import CrossEntropy
from .mse import MSE
from .mae import MAE
from .huber import Huber

LOSSES = {

    "cross_entropy": CrossEntropy,

    "mse": MSE,

    "mae": MAE,

    "huber": Huber
}