import matplotlib.pyplot as plt


def plot_top10_accuracy(df):

    top10 = (
        df.sort_values(
            by="final_accuracy",
            ascending=False
        )
        .head(10)
    )

    labels = [

        f"{o}\n{a}"

        for o, a in zip(
            top10["optimizer"],
            top10["activation"]
        )
    ]

    plt.figure(figsize=(14, 8))

    plt.bar(
        labels,
        top10["final_accuracy"]
    )

    plt.xticks(rotation=45)

    plt.ylabel("Accuracy")

    plt.title(
        "Top 10 Configurations"
    )

    plt.tight_layout()

    plt.savefig(
        "results/top10_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_optimizer_comparison(df):

    grouped = (
        df.groupby("optimizer")
        ["final_accuracy"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    grouped.plot(kind="bar")

    plt.ylabel("Accuracy")

    plt.title(
        "Optimizer Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        "results/optimizer_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_activation_comparison(df):

    grouped = (
        df.groupby("activation")
        ["final_accuracy"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    grouped.plot(kind="bar")

    plt.ylabel("Accuracy")

    plt.title(
        "Activation Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        "results/activation_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_initializer_comparison(df):

    grouped = (
        df.groupby("initializer")
        ["final_accuracy"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    grouped.plot(kind="bar")

    plt.ylabel("Accuracy")

    plt.title(
        "Initializer Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        "results/initializer_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()