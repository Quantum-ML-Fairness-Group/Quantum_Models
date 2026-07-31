import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_results(csv_path="results.csv"):
    df = pd.read_csv(csv_path)

    required_cols = ["model", "accuracy", "demographic_parity_difference"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only the 50-epoch results
    epoch_50_models = ["VQC", "QNN", "CCQC"]
    epoch_50_values = {
        "VQC": 0.6699187159538269,
        "QNN": 0.6650406718254089,
        "CCQC": 0.639837384223938,
    }

    df = df[
        (df["model"].isin(epoch_50_models)) &
        (df["accuracy"].round(10).isin([round(v, 10) for v in epoch_50_values.values()]))
    ]

    models = df["model"]
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, df["accuracy"], width, label="Accuracy")
    plt.bar(
        x + width / 2,
        df["demographic_parity_difference"],
        width,
        label="Demographic Parity Difference"
    )

    plt.title("50 Epochs: Accuracy vs Demographic Parity Difference")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("epoch_50_combined_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_results("results.csv")