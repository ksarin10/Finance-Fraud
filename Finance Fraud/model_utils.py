"""Shared data loading, preprocessing, and reporting utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_PATH = Path(__file__).with_name("creditcardcsvpresent.csv")
NUMERICAL_FEATURES = [
    "Average Amount/transaction/day",
    "Transaction_amount",
    "Total Number of declines/day",
    "Daily_chargeback_avg_amt",
    "6_month_avg_chbk_amt",
    "6-month_chbk_freq",
]
CATEGORICAL_FEATURES = ["isForeignTransaction", "isHighRiskCountry"]
TARGET = "isFradulent"  # Original dataset spelling retained for compatibility.
EXCLUDED_COLUMNS = ["Merchant_id", "Transaction date", TARGET, "Is declined"]


def parse_data_argument(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV dataset path (default: {DEFAULT_DATA_PATH.name})",
    )
    return parser.parse_args()


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_split(data_path: Path):
    df = pd.read_csv(data_path)
    required = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + EXCLUDED_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    labels = df[TARGET].map({"N": 0, "Y": 1})
    if labels.isna().any():
        invalid = sorted(df.loc[labels.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Unexpected fraud labels: {', '.join(invalid)}")

    features = df.drop(columns=EXCLUDED_COLUMNS)
    return train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )


def print_evaluation(name: str, actual, predicted) -> None:
    print(f"\n{name}")
    print(classification_report(actual, predicted, digits=2, zero_division=0))
    print("Confusion matrix")
    print(confusion_matrix(actual, predicted))
