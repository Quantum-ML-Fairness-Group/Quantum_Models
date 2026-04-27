"""
Dataset-variability using VQC base code

Fixed condition:
  - architecture: VQC
  - encoding: angle encoding (the VQC default)
  - noise: none (default.qubit simulator)
  - fairness metric: demographic parity difference

"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import CompasDataset, make_compas_dataloaders
from evaluate import evaluate_model
from model_architectures.vqc import VQC
from train import train_model


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
QML_BIAS_AUDIT_ROOT = WORKSPACE_ROOT / "qml-bias-audit"
if QML_BIAS_AUDIT_ROOT.exists():
    sys.path.insert(0, str(QML_BIAS_AUDIT_ROOT))

from data.registry import get_dataset_loader, list_datasets  # noqa: E402


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    feature_names: list[str]
    resolved_attribute: str
    n_train: int
    n_val: int
    n_test: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_attribute(groups: dict[str, np.ndarray], requested: str) -> str:
    if requested != "auto":
        if requested not in groups:
            available = ", ".join(groups.keys())
            raise ValueError(
                f"Requested protected attribute {requested!r} is not available. "
                f"Available: {available}"
            )
        return requested
    if "race" in groups:
        return "race"
    if "sex" in groups:
        return "sex"
    return next(iter(groups.keys()))


def sample_indices(
    n_total: int,
    n_keep: int | None,
    seed: int,
    groups: dict[str, np.ndarray] | None = None,
) -> np.ndarray | None:
    if n_keep is None or n_keep >= n_total:
        return None

    rng = np.random.RandomState(seed)
    if groups:
        primary = groups["race"] if "race" in groups else next(iter(groups.values()))
        required = []
        for value in np.unique(primary):
            candidates = np.flatnonzero(primary == value)
            if len(candidates) > 0:
                required.append(rng.choice(candidates))
        required = np.unique(required)
        if len(required) < n_keep:
            remaining = np.setdiff1d(np.arange(n_total), required, assume_unique=False)
            fill = rng.choice(remaining, size=n_keep - len(required), replace=False)
            return np.concatenate([required, fill])
        return required[:n_keep]

    return rng.choice(n_total, size=n_keep, replace=False)


def subsample_split(
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
    n_keep: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = sample_indices(len(X), n_keep, seed, {"sensitive": sensitive})
    if idx is None:
        return X, y, sensitive
    return X[idx], y[idx], sensitive[idx]


def local_data_dir(dataset: str) -> Path | None:
    csvs = WORKSPACE_ROOT / "csvs"
    candidates = {
        "adult": csvs / "adult (1)",
        "cardiovascular": csvs,
        "diabetes_hospital": csvs / "archive (3)",
        "diabetes_prediction": csvs,
        "glioma": csvs / "glioma+grading+clinical+and+mutation+features+dataset",
        "heart_indicators": csvs / "archive (2)" / "2020",
    }
    path = candidates.get(dataset)
    return path if path is not None and path.exists() else None


def tensor_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sensitive: np.ndarray,
) -> CompasDataset:
    return CompasDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).view(-1, 1),
        torch.tensor(sensitive, dtype=torch.float32).view(-1, 1),
    )


def bundle_from_splits(args: argparse.Namespace, dataset: str, splits: dict) -> DatasetBundle:
    attribute = resolve_attribute(splits["groups_test"], args.attribute)

    X_train, y_train, s_train = subsample_split(
        splits["X_train"],
        splits["y_train"],
        splits["groups_train"][attribute],
        args.n_train,
        args.seed,
    )
    X_val, y_val, s_val = subsample_split(
        splits["X_val"],
        splits["y_val"],
        splits["groups_val"][attribute],
        args.n_val,
        args.seed,
    )
    X_test, y_test, s_test = subsample_split(
        splits["X_test"],
        splits["y_test"],
        splits["groups_test"][attribute],
        args.n_test,
        args.seed,
    )

    return DatasetBundle(
        train_loader=DataLoader(
            tensor_dataset(X_train, y_train, s_train),
            batch_size=args.batch_size,
            shuffle=True,
        ),
        val_loader=DataLoader(
            tensor_dataset(X_val, y_val, s_val),
            batch_size=args.batch_size,
            shuffle=False,
        ),
        test_loader=DataLoader(
            tensor_dataset(X_test, y_test, s_test),
            batch_size=args.batch_size,
            shuffle=False,
        ),
        input_dim=X_train.shape[1],
        feature_names=list(splits["feature_names"]),
        resolved_attribute=attribute,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
    )


def load_dataset_bundle(args: argparse.Namespace, dataset: str) -> DatasetBundle:
    if dataset == "compas":
        compas_csv = ROOT / "compas-scores-two-years.csv"
        bundle = make_compas_dataloaders(
            str(compas_csv),
            batch_size=args.batch_size,
            test_size=0.2,
            random_state=args.seed,
        )

        X_train, y_train, s_train = subsample_split(
            bundle.X_train.numpy(),
            bundle.y_train.view(-1).numpy(),
            bundle.sensitive_train.view(-1).numpy(),
            args.n_train,
            args.seed,
        )
        X_test, y_test, s_test = subsample_split(
            bundle.X_test.numpy(),
            bundle.y_test.view(-1).numpy(),
            bundle.sensitive_test.view(-1).numpy(),
            args.n_test,
            args.seed,
        )
        # The teammate COMPAS loader has no separate validation split, so use a
        # fixed subset of the training split for validation monitoring.
        X_val, y_val, s_val = subsample_split(X_train, y_train, s_train, args.n_val, args.seed)

        return DatasetBundle(
            train_loader=DataLoader(
                tensor_dataset(X_train, y_train, s_train),
                batch_size=args.batch_size,
                shuffle=True,
            ),
            val_loader=DataLoader(
                tensor_dataset(X_val, y_val, s_val),
                batch_size=args.batch_size,
                shuffle=False,
            ),
            test_loader=DataLoader(
                tensor_dataset(X_test, y_test, s_test),
                batch_size=args.batch_size,
                shuffle=False,
            ),
            input_dim=X_train.shape[1],
            feature_names=bundle.feature_names,
            resolved_attribute="race",
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
        )

    loader = get_dataset_loader(dataset)
    kwargs = {"random_state": args.seed}
    data_dir = local_data_dir(dataset)
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    splits = loader(**kwargs)
    return bundle_from_splits(args, dataset, splits)


def run_one_dataset(args: argparse.Namespace, dataset: str) -> dict:
    set_seed(args.seed)
    bundle = load_dataset_bundle(args, dataset)

    model = VQC(
        input_dim=bundle.input_dim,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        output_dim=1,
    )

    print(
        f"\n[{dataset}] VQC | angle encoding | no noise | "
        f"features={bundle.input_dim} | qubits={args.n_qubits} | "
        f"layers={args.n_layers} | train={bundle.n_train}"
    )

    t0 = time.time()
    train_model(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        epochs=args.epochs,
        lr=args.lr,
    )
    train_time = time.time() - t0

    results = evaluate_model(model=model, data_loader=bundle.test_loader)
    return {
        "dataset": dataset,
        "model": "VQC",
        "encoding": "angle",
        "noise": "none",
        "fairness_metric": "demographic_parity_difference",
        "resolved_attribute": bundle.resolved_attribute,
        "input_dim": bundle.input_dim,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "n_train": bundle.n_train,
        "n_val": bundle.n_val,
        "n_test": bundle.n_test,
        "train_time_s": train_time,
        "accuracy": results["accuracy"],
        "demographic_parity_difference": results["demographic_parity_difference"],
    }


def save_plot(df: pd.DataFrame) -> Path:
    plot_df = df.sort_values("dataset")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ["#3B6EA8", "#D95F02", "#2E8B57", "#8A63A8", "#C44E52", "#4C8C8A", "#8C6D31"]
    bars = ax.bar(
        plot_df["dataset"],
        plot_df["demographic_parity_difference"],
        color=colors[: len(plot_df)],
        edgecolor="white",
    )

    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1, label="0.1 threshold")
    ax.set_ylabel("Demographic Parity Difference (lower is fairer)")
    ax.set_xlabel("Dataset")
    ax.set_title("Dataset Variability with Fixed VQC Angle Encoding")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", labelrotation=25)

    for bar, acc, attr in zip(
        bars,
        plot_df["accuracy"],
        plot_df["resolved_attribute"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{attr}\nacc={acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ymax = max(0.25, float(plot_df["demographic_parity_difference"].max()) * 1.25)
    ax.set_ylim(0, ymax)
    plt.tight_layout()

    out = FIGURES_DIR / "dataset_variability_vqc_angle_dpd.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main(args: argparse.Namespace) -> None:
    unavailable = sorted(set(args.datasets) - set(list_datasets()))
    if unavailable:
        raise ValueError(f"Unknown dataset(s): {unavailable}. Available: {list_datasets()}")

    rows = [run_one_dataset(args, dataset) for dataset in args.datasets]
    df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "dataset_variability_vqc_angle_dpd.csv"
    json_path = RESULTS_DIR / "dataset_variability_vqc_angle_dpd.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    fig_path = save_plot(df)
    summary_cols = [
        "dataset",
        "accuracy",
        "demographic_parity_difference",
        "resolved_attribute",
        "n_train",
        "n_val",
        "n_test",
        "train_time_s",
    ]
    print("\nDataset variability summary:")
    print(df[summary_cols].round(4).to_string(index=False))
    print(f"\nResults saved to {csv_path}")
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["compas", "adult"],
        choices=list_datasets(),
        help="Datasets to compare.",
    )
    parser.add_argument(
        "--attribute",
        default="auto",
        help="Protected attribute to evaluate, or 'auto' for race when available else sex.",
    )
    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_val", type=int, default=300)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
