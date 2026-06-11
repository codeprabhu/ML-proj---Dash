from metrics import accuracy
from core.network import NeuralNetworks

def run_experiment(
    X_train,
    y_train,
    initializer,
    hidden_activation,
    output_activation,
    loss,
    optimizer_class,
    epochs=20
):

    network = NeuralNetworks(
        layer_sizes=[784,128,64,10],

        hidden_activation=hidden_activation,

        output_activation=output_activation,

        initializer=initializer
    )

    optimizer = optimizer_class(
        network.parameters
    )

    history = {
        "loss": [],
        "accuracy": []
    }

    for epoch in range(epochs):

        Y_hat, cache = network.forward(
            X_train
        )

        loss_value = loss.forward(
            y_train,
            Y_hat
        )

        grads = network.backward(
            X_train,
            y_train,
            cache,
            loss
        )

        optimizer.step(
            network.parameters,
            grads
        )

        acc = accuracy(
            y_train,
            Y_hat
        )

        history["loss"].append(
            loss_value
        )

        history["accuracy"].append(
            acc
        )

    return {
        "loss": history["loss"],
        "accuracy": history["accuracy"],
        "final_loss": history["loss"][-1],
        "final_accuracy": history["accuracy"][-1]
    }