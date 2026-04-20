import csv
import os


def save_results_to_csv(
    file_path: str,
    model_name: str,
    accuracy: float,
    dpd: float,
):
    """
    Append experiment results to a CSV file.

    If file does not exist, create it with header.
    """

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(["model", "accuracy", "demographic_parity_difference"])

        writer.writerow([model_name, accuracy, dpd])