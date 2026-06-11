import os

os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
import time

start_time = time.time()
import pandas as pd # type: ignore

from core.data import (
    load_data,
    preprocess_data,
    one_hot_encode
)

from core.experiment import run_experiment

from plotting import (
    plot_top10_accuracy,
    plot_optimizer_comparison,
    plot_activation_comparison,
    plot_initializer_comparison
)

from core.initialization import (
    RandomNormal,
    Xavier,
    He,
    LeCun
)

from activations.relu import ReLU
from activations.leaky_relu import LeakyReLU
from activations.prelu import PReLU
from activations.elu import ELU
from activations.swish import Swish
from activations.gelu import GELU
from activations.mish import Mish
from activations.sigmoid import Sigmoid
from activations.tanh import Tanh
from activations.softmax import Softmax

from losses.cross_entropy import CrossEntropy
from losses.mse import MSE
from losses.mae import MAE
from losses.huber import Huber

from optimizer.sgd import SGD
from optimizer.momentum import Momentum
from optimizer.nesterov import Nesterov
from optimizer.adagrad import AdaGrad
from optimizer.rmsprop import RMSProp
from optimizer.adam import Adam


# =====================================================
# Setup
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)

print("Loading dataset...")

X_train, X_test, y_train, y_test = load_data()

X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# -----------------------------------------------------
# DEBUG MODE
# -----------------------------------------------------
# Uncomment for fast testing
#
# X_train = X_train[:5000]
# y_train = y_train[:5000]
#
# -----------------------------------------------------


# =====================================================
# Search Space
# =====================================================

optimizers = {

    "SGD": SGD,

    "Momentum": Momentum,

    "Nesterov": Nesterov,

    "AdaGrad": AdaGrad,

    "RMSProp": RMSProp,

    "Adam": Adam
}

activations = {

    "ReLU": ReLU(),

    "LeakyReLU": LeakyReLU(),

    "PReLU": PReLU(),

    "ELU": ELU(),

    "Swish": Swish(),

    "GELU": GELU(),

    "Mish": Mish(),

    "Sigmoid": Sigmoid(),

    "Tanh": Tanh()
}

initializers = {

    "Random": RandomNormal(),

    "Xavier": Xavier(),

    "He": He(),

    "LeCun": LeCun()
}

losses = {

    "CrossEntropy": CrossEntropy(),

    "MSE": MSE(),

    "MAE": MAE(),

    "Huber": Huber()
}


# =====================================================
# Experiment Loop
# =====================================================

results = []

total_runs = (
    len(optimizers)
    * len(activations)
    * len(initializers)
    * len(losses)
)

current_run = 0

print(f"\nTotal Experiments: {total_runs}\n")

for optimizer_name, optimizer_class in optimizers.items():

    for activation_name, activation in activations.items():

        for initializer_name, initializer in initializers.items():

            for loss_name, loss in losses.items():

                current_run += 1

                print(
                    f"[{current_run}/{total_runs}] "
                    f"{optimizer_name} | "
                    f"{activation_name} | "
                    f"{initializer_name} | "
                    f"{loss_name}"
                )

                try:

                    history = run_experiment(

                        X_train,
                        y_train,

                        initializer,

                        activation,

                        Softmax(),

                        loss,

                        optimizer_class,

                        epochs=20
                    )

                    results.append({

                        "optimizer":
                            optimizer_name,

                        "activation":
                            activation_name,

                        "initializer":
                            initializer_name,

                        "loss":
                            loss_name,

                        "final_loss":
                            history["final_loss"],

                        "final_accuracy":
                            history["final_accuracy"]
                    })

                except Exception as e:

                    print(
                        f"FAILED: "
                        f"{optimizer_name} | "
                        f"{activation_name} | "
                        f"{initializer_name} | "
                        f"{loss_name}"
                    )

                    print(e)

                    continue


# =====================================================
# Save Results
# =====================================================

df = pd.DataFrame(results)

df.to_csv(
    "results/all_results.csv",
    index=False
)

print("\nSaved all_results.csv")


# =====================================================
# Top 10
# =====================================================

top10 = (
    df.sort_values(
        by="final_accuracy",
        ascending=False
    )
    .head(10)
)

top10.to_csv(
    "results/top10_configs.csv",
    index=False
)

print("\nTop 10 Configurations:\n")

print(top10)


# =====================================================
# Best Configuration
# =====================================================

best = (
    df.sort_values(
        by="final_accuracy",
        ascending=False
    )
    .iloc[0]
)

print("\nBEST CONFIGURATION\n")

print(best)


# =====================================================
# Graphs
# =====================================================

print("\nGenerating graphs...\n")

plot_top10_accuracy(df)

plot_optimizer_comparison(df)

plot_activation_comparison(df)

plot_initializer_comparison(df)

print("\nDone.")

print(
    f"\nRuntime: "
    f"{(time.time() - start_time)/60:.2f} minutes"
)