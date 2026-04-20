from typing import Dict, Optional

import torch

from fairness import (
    binarize_predictions,
    demographic_parity_difference,
)


@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run the model on a dataset and collect outputs.

    Args:
        model: trained model (VQC / QNN / CCQC)
        data_loader: DataLoader returning (x, y, sensitive)
        device: "cpu" or "cuda"; if None, auto-select

    Returns:
        Dictionary containing:
            - logits
            - preds
            - labels
            - sensitive
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_logits = []
    all_preds = []
    all_labels = []
    all_sensitive = []

    for x, y, sensitive in data_loader:
        x = x.to(device)
        y = y.to(device)
        sensitive = sensitive.to(device)

        logits, _ = model(x)

        logits = logits.view(-1)
        preds = binarize_predictions(logits)

        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.view(-1).cpu().int())
        all_sensitive.append(sensitive.view(-1).cpu().int())

    return {
        "logits": torch.cat(all_logits, dim=0),
        "preds": torch.cat(all_preds, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "sensitive": torch.cat(all_sensitive, dim=0),
    }


def compute_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.

    Args:
        y_pred: predicted binary labels, shape (N,)
        y_true: true binary labels, shape (N,)

    Returns:
        Accuracy as a float.
    """
    y_pred = y_pred.view(-1).int()
    y_true = y_true.view(-1).int()

    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length.")

    return (y_pred == y_true).float().mean().item()


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained model on a dataset.

    Computes:
        - accuracy
        - demographic parity difference

    Args:
        model: trained model
        data_loader: DataLoader returning (x, y, sensitive)
        device: "cpu" or "cuda"; if None, auto-select

    Returns:
        Dictionary with evaluation results.
    """
    outputs = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    accuracy = compute_accuracy(outputs["preds"], outputs["labels"])
    dpd = demographic_parity_difference(
        y_pred=outputs["preds"],
        sensitive=outputs["sensitive"],
        absolute=True,
    ).item()

    return {
        "accuracy": accuracy,
        "demographic_parity_difference": dpd,
    }


if __name__ == "__main__":
    print(
        "This file provides evaluation utilities.\n"
        "Use evaluate_model(model, data_loader) after training."
    )