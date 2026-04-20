from typing import Dict, List, Optional

import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        model: quantum model (e.g. VQC)
        train_loader: PyTorch DataLoader returning (x, y, sensitive)
        optimizer: optimizer instance
        loss_fn: loss function, e.g. BCEWithLogitsLoss
        device: "cpu" or "cuda"

    Returns:
        Dictionary with average epoch loss and accuracy.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)

        # Ensure shapes match for BCE loss
        logits = logits.view(-1, 1)
        y = y.view(-1, 1)

        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    data_loader,
    loss_fn: nn.Module,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate the model for one epoch on validation/test data.

    Args:
        model: quantum model
        data_loader: PyTorch DataLoader returning (x, y, sensitive)
        loss_fn: loss function
        device: "cpu" or "cuda"

    Returns:
        Dictionary with average loss and accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)

        logits = logits.view(-1, 1)
        y = y.view(-1, 1)

        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Full training loop for a quantum model.

    Args:
        model: VQC / QNN / CCQC model
        train_loader: training DataLoader
        val_loader: optional validation/test DataLoader
        epochs: number of training epochs
        lr: learning rate
        device: "cpu" or "cuda"; if None, auto-select

    Returns:
        Training history dictionary.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
    }

    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])

        if val_loader is not None:
            val_metrics = validate_one_epoch(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )

    return history