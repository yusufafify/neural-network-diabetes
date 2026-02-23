"""
dataloader.py — Data Loading & Splitting for the Diabetes Pipeline

Loads the pre-processed diabetes_cleaned.csv, separates features from
the target label ('Outcome'), and performs a stratified 80/20
train-test split.  Features are already StandardScaled so no
additional normalization is needed here.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Configuration ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "diabetes_cleaned.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COL = "Outcome"


def load_data(
    data_path: str = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Load the cleaned diabetes dataset and split it for training.

    Parameters
    ----------
    data_path : str
        Path to the CSV file (default: ``../dataset/diabetes_cleaned.csv``).
    test_size : float
        Fraction of the data reserved for testing (default: 0.2).
    random_state : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    X_train : np.ndarray
        Training features, shape (n_train, n_features).
    X_test : np.ndarray
        Test features, shape (n_test, n_features).
    y_train : np.ndarray
        Training labels, shape (n_train,).
    y_test : np.ndarray
        Test labels, shape (n_test,).
    """
    # Load CSV
    df = pd.read_csv(data_path)

    # Separate features and target
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def get_num_features(data_path: str = DATA_PATH) -> int:
    """Return the number of feature columns in the dataset."""
    df = pd.read_csv(data_path, nrows=0)
    return len([c for c in df.columns if c != TARGET_COL])
