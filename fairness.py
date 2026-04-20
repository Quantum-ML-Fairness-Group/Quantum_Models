import torch


def binarize_predictions(outputs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert model outputs into binary predictions.

    Assumes outputs are logits or probabilities of shape (N,) or (N, 1).

    If outputs are logits, applies sigmoid first.
    If outputs are already probabilities in [0, 1], sigmoid will still work,
    but ideally pass logits consistently from the model.

    Returns:
        Tensor of shape (N,) with values in {0, 1}.
    """
    if outputs.ndim > 2:
        raise ValueError("outputs must have shape (N,) or (N, 1).")

    outputs = outputs.view(-1)

    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).int()

    return preds


def positive_prediction_rate(
    y_pred: torch.Tensor,
    sensitive: torch.Tensor,
    group_value: int,
) -> torch.Tensor:
    """
    Compute P(y_hat = 1 | A = group_value)

    Args:
        y_pred: binary predictions, shape (N,) or (N,1)
        sensitive: binary sensitive attribute, shape (N,) or (N,1)
        group_value: group to evaluate (0 or 1)

    Returns:
        Scalar tensor containing the positive prediction rate for that group.
    """
    y_pred = y_pred.view(-1).int()
    sensitive = sensitive.view(-1).int()

    mask = sensitive == group_value

    if mask.sum() == 0:
        raise ValueError(f"No samples found for sensitive group {group_value}.")

    return y_pred[mask].float().mean()


def demographic_parity_difference(
    y_pred: torch.Tensor,
    sensitive: torch.Tensor,
    absolute: bool = True,
) -> torch.Tensor:
    """
    Compute demographic parity difference.

    Formula:
        P(y_hat = 1 | A = 0) - P(y_hat = 1 | A = 1)

    Args:
        y_pred: binary predictions, shape (N,) or (N,1)
        sensitive: binary sensitive attribute, shape (N,) or (N,1)
                   expected values are 0 and 1
        absolute: if True, return absolute value

    Returns:
        Scalar tensor for demographic parity difference.
    """
    rate_group_0 = positive_prediction_rate(y_pred, sensitive, group_value=0)
    rate_group_1 = positive_prediction_rate(y_pred, sensitive, group_value=1)

    dpd = rate_group_0 - rate_group_1

    if absolute:
        dpd = torch.abs(dpd)

    return dpd


def demographic_parity_from_logits(
    logits: torch.Tensor,
    sensitive: torch.Tensor,
    threshold: float = 0.5,
    absolute: bool = True,
) -> torch.Tensor:
    """
    Convenience function:
    Convert logits to binary predictions, then compute demographic parity difference.

    Args:
        logits: model outputs, shape (N,) or (N,1)
        sensitive: binary sensitive attribute, shape (N,) or (N,1)
        threshold: threshold after sigmoid
        absolute: whether to return absolute DPD

    Returns:
        Scalar tensor for demographic parity difference.
    """
    y_pred = binarize_predictions(logits, threshold=threshold)
    return demographic_parity_difference(y_pred, sensitive, absolute=absolute)


if __name__ == "__main__":
    # Example usage
    logits = torch.tensor([[0.8], [-1.2], [1.5], [0.1], [-0.7], [2.0]])
    sensitive = torch.tensor([[0], [0], [1], [1], [0], [1]])

    preds = binarize_predictions(logits)
    dpd = demographic_parity_difference(preds, sensitive)
    dpd_from_logits = demographic_parity_from_logits(logits, sensitive)

    print("Predictions:", preds.tolist())
    print("Demographic Parity Difference:", dpd.item())
    print("DPD from logits:", dpd_from_logits.item())