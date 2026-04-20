import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class CompasDataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    feature_names: List[str]
    scaler: StandardScaler
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    sensitive_train: torch.Tensor
    sensitive_test: torch.Tensor


class CompasDataset(Dataset):
    """
    PyTorch dataset for COMPAS experiments.

    Returns:
        x: feature tensor
        y: target tensor
        sensitive: sensitive attribute tensor
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sensitive: torch.Tensor,
    ):
        if not (len(X) == len(y) == len(sensitive)):
            raise ValueError("X, y, and sensitive must have the same length.")

        self.X = X.float()
        self.y = y.float()
        self.sensitive = sensitive.float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.sensitive[idx]


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in COMPAS CSV: {missing}"
        )


def _preprocess_compas_dataframe(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess COMPAS dataframe for a fairness experiment.

    Target:
        two_year_recid

    Sensitive attribute:
        race
        Encoded as:
            1 -> African-American
            0 -> Caucasian

    Notes:
    - Keeps only African-American and Caucasian rows
    - Uses a common subset of structured features
    - One-hot encodes categorical variables
    """

    required_columns = [
        "sex",
        "age",
        "age_cat",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]
    _validate_columns(df, required_columns)

    # Keep only the two race groups used in the fairness comparison
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()

    if df.empty:
        raise ValueError(
            "No rows left after filtering race to African-American/Caucasian."
        )

    # Optional NA cleanup
    if drop_na:
        df = df.dropna(subset=required_columns).copy()

    # Target label
    y = df["two_year_recid"].astype(int)

    # Sensitive attribute:
    # 1 = African-American, 0 = Caucasian
    sensitive = (df["race"] == "African-American").astype(int)

    # Features to use
    feature_columns_numeric = [
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
    ]

    feature_columns_categorical = [
        "sex",
        "age_cat",
        "c_charge_degree",
    ]

    features_df = df[feature_columns_numeric + feature_columns_categorical].copy()

    # One-hot encode categorical features
    features_df = pd.get_dummies(
        features_df,
        columns=feature_columns_categorical,
        drop_first=True,
        dtype=np.float32,
    )

    # Ensure numeric features are numeric
    for col in feature_columns_numeric:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    if drop_na:
        valid_mask = features_df.notna().all(axis=1)
        features_df = features_df.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()
        sensitive = sensitive.loc[valid_mask].copy()

    return features_df, y, sensitive


def load_compas_dataframe(
    csv_path: str,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess COMPAS CSV.

    Args:
        csv_path: path to COMPAS CSV file
        drop_na: whether to drop rows with missing required values

    Returns:
        X_df, y, sensitive
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"COMPAS CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return _preprocess_compas_dataframe(df, drop_na=drop_na)


def make_compas_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by_target: bool = True,
    drop_na: bool = True,
    scale_features: bool = True,
    shuffle_train: bool = True,
) -> CompasDataBundle:
    """
    Build train/test DataLoaders for COMPAS.

    Args:
        csv_path: path to COMPAS CSV
        batch_size: batch size for DataLoaders
        test_size: fraction used for test split
        random_state: reproducibility seed
        stratify_by_target: whether to stratify train/test split by y
        drop_na: whether to drop NA rows in required columns
        scale_features: whether to standardize X using StandardScaler
        shuffle_train: whether to shuffle training DataLoader

    Returns:
        CompasDataBundle containing DataLoaders and metadata
    """
    X_df, y, sensitive = load_compas_dataframe(
        csv_path=csv_path,
        drop_na=drop_na,
    )

    feature_names = list(X_df.columns)

    X_train_df, X_test_df, y_train_s, y_test_s, s_train_s, s_test_s = train_test_split(
        X_df,
        y,
        sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify_by_target else None,
    )

    scaler = StandardScaler()
    if scale_features:
        X_train_np = scaler.fit_transform(X_train_df).astype(np.float32)
        X_test_np = scaler.transform(X_test_df).astype(np.float32)
    else:
        # Fit scaler anyway so the return object is always consistent
        scaler.fit(X_train_df)
        X_train_np = X_train_df.to_numpy(dtype=np.float32)
        X_test_np = X_test_df.to_numpy(dtype=np.float32)

    y_train_np = y_train_s.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_np = y_test_s.to_numpy(dtype=np.float32).reshape(-1, 1)

    s_train_np = s_train_s.to_numpy(dtype=np.float32).reshape(-1, 1)
    s_test_np = s_test_s.to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    sensitive_train = torch.tensor(s_train_np, dtype=torch.float32)
    sensitive_test = torch.tensor(s_test_np, dtype=torch.float32)

    train_dataset = CompasDataset(X_train, y_train, sensitive_train)
    test_dataset = CompasDataset(X_test, y_test, sensitive_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return CompasDataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=X_train.shape[1],
        feature_names=feature_names,
        scaler=scaler,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
    )


if __name__ == "__main__":
    # Example usage:
    #
    # Replace with your actual CSV path
    csv_path = "compas.csv"

    bundle = make_compas_dataloaders(
        csv_path=csv_path,
        batch_size=32,
        test_size=0.2,
        random_state=42,
    )

    print("COMPAS loaded successfully.")
    print(f"Input dimension: {bundle.input_dim}")
    print(f"Train size: {len(bundle.X_train)}")
    print(f"Test size: {len(bundle.X_test)}")
    print(f"Sensitive attribute: race (1 = African-American, 0 = Caucasian)")
    print(f"Target label: two_year_recid")