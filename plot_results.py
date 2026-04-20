import pandas as pd
import matplotlib.pyplot as plt


def plot_results(csv_path="results.csv"):
    df = pd.read_csv(csv_path)

    required_cols = ["model", "accuracy", "demographic_parity_difference"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["accuracy"])
    plt.title("Accuracy Across Quantum Model Architectures")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()

    # Demographic parity difference plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["demographic_parity_difference"])
    plt.title("Demographic Parity Difference Across Quantum Model Architectures")
    plt.xlabel("Model")
    plt.ylabel("Demographic Parity Difference")
    plt.tight_layout()
    plt.savefig("dpd_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_results("results.csv")