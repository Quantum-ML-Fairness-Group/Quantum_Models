import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


@dataclass
class AdultDataBundle:
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


class AdultDataset(Dataset):
    def __init__(self, X, y, sensitive):
        if not (len(X) == len(y) == len(sensitive)):
            raise ValueError("X, y, and sensitive must have same length.")

        self.X = X.float()
        self.y = y.float()
        self.sensitive = sensitive.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sensitive[idx]


def _clean_adult_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip spaces from column names and string values
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # adult.test sometimes has labels like ">50K."
    df["Target"] = df["Target"].str.replace(".", "", regex=False)

    # Replace missing values marked as ?
    df = df.replace("?", np.nan)
    df = df.dropna().copy()

    return df


def _preprocess_adult(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    required_columns = [
        "Age",
        "Workclass",
        "fnlwgt",
        "Education",
        "Education_Num",
        "Martial_Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Capital_Gain",
        "Capital_Loss",
        "Hours_per_week",
        "Country",
        "Target",
    ]

    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Missing column in train data: {col}")
        if col not in test_df.columns:
            raise ValueError(f"Missing column in test data: {col}")

    train_df = _clean_adult_dataframe(train_df)
    test_df = _clean_adult_dataframe(test_df)

    # Target: 1 = income >50K, 0 = income <=50K
    y_train = (train_df["Target"] == ">50K").astype(int)
    y_test = (test_df["Target"] == ">50K").astype(int)

    # Sensitive attribute:
    # 1 = Male, 0 = Female
    sensitive_train = (train_df["Sex"] == "Male").astype(int)
    sensitive_test = (test_df["Sex"] == "Male").astype(int)

    # Drop target from features
    X_train_df = train_df.drop(columns=["Target"]).copy()
    X_test_df = test_df.drop(columns=["Target"]).copy()

    # IMPORTANT:
    # Drop Sex from features because it is the sensitive attribute.
    # Keep Race as a regular feature for now unless your mentor says otherwise.
    X_train_df = X_train_df.drop(columns=["Sex"])
    X_test_df = X_test_df.drop(columns=["Sex"])

    # Combine temporarily so one-hot columns match between train and test
    combined = pd.concat([X_train_df, X_test_df], axis=0)

    categorical_cols = combined.select_dtypes(include="object").columns.tolist()

    combined = pd.get_dummies(
        combined,
        columns=categorical_cols,
        drop_first=True,
        dtype=np.float32,
    )

    X_train_encoded = combined.iloc[: len(X_train_df)].copy()
    X_test_encoded = combined.iloc[len(X_train_df):].copy()

    feature_names = list(X_train_encoded.columns)

    return (
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        sensitive_train,
        sensitive_test,
        feature_names,
    )


def make_adult_dataloaders(
    train_csv_path: str = "adult_train.csv",
    test_csv_path: str = "adult_test.csv",
    batch_size: int = 32,
    scale_features: bool = True,
    shuffle_train: bool = True,
) -> AdultDataBundle:

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Adult train CSV not found at: {train_csv_path}")

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Adult test CSV not found at: {test_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    (
        X_train_df,
        X_test_df,
        y_train_s,
        y_test_s,
        s_train_s,
        s_test_s,
        feature_names,
    ) = _preprocess_adult(train_df, test_df)

    scaler = StandardScaler()

    if scale_features:
        X_train_np = scaler.fit_transform(X_train_df).astype(np.float32)
        X_test_np = scaler.transform(X_test_df).astype(np.float32)
    else:
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

    train_dataset = AdultDataset(X_train, y_train, sensitive_train)
    test_dataset = AdultDataset(X_test, y_test, sensitive_test)

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

    return AdultDataBundle(
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
    bundle = make_adult_dataloaders(
        train_csv_path="adult_train.csv",
        test_csv_path="adult_test.csv",
        batch_size=32,
    )

    print("Adult dataset loaded successfully.")
    print(f"Input dimension: {bundle.input_dim}")
    print(f"Train size: {len(bundle.X_train)}")
    print(f"Test size: {len(bundle.X_test)}")
    print("Sensitive attribute: Sex (1 = Male, 0 = Female)")
    print("Target label: income >50K")