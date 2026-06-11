import numpy as np
from sklearn.datasets import fetch_openml # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

def load_data():
    mnist = fetch_openml("mnist_784", version = 1, as_frame=False)
    x = mnist.data
    y = mnist.target.astype(int)

    return train_test_split(x, y, test_size=10000, random_state=42)

def preprocess_data(X):
    X = X.astype(np.float32)
    X /= 255.0
    return X
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]