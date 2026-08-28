"""
data_prep.py
------------
STEP 1-5 of the pipeline: load raw CSV, clean, encode, define
X / y_binary / y_multi, and produce the train/test split used by
every downstream script.

Run this first. It writes data/processed/split_data.joblib, which
model1_stage1_training.py, model2_stage2_training.py,
compare_models.py, and cascade_pipeline.py all load from — this is
the "connection" between the files, so every script trains and
evaluates on the exact same rows.

Usage:
    python src/data_prep.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV_PATH = os.path.join("data", "ai4i2020.csv")
OUT_PATH = os.path.join("data", "processed", "split_data.joblib")

FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Warning — missing values found:")
        print(missing[missing > 0])
    else:
        print("No missing values.")

    # Drop identifier columns — not predictive, would leak row identity
    df = df.drop(columns=["UDI", "Product ID"])
    return df


def prepare_features_and_targets(df: pd.DataFrame):
    print("Encoding the 'Type' column (L, M, H)...")
    df_encoded = pd.get_dummies(df, columns=["Type"], drop_first=True, dtype=int)

    y_binary = df_encoded["Machine failure"]
    y_multi = df_encoded[FAILURE_TYPES]

    columns_to_drop = ["Machine failure"] + FAILURE_TYPES
    X = df_encoded.drop(columns=columns_to_drop)

    return X, y_binary, y_multi


def split(X, y_binary, y_multi, test_size=0.2, random_state=42):
    print(f"Splitting data ({int((1-test_size)*100)}/{int(test_size*100)}) with stratification...")
    X_bi_train, X_bi_test, y_bi_train, y_bi_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    y_multi_train = y_multi.loc[X_bi_train.index]
    y_multi_test = y_multi.loc[X_bi_test.index]

    assert (X_bi_train.index == y_multi_train.index).all()
    assert (X_bi_test.index == y_multi_test.index).all()

    return X_bi_train, X_bi_test, y_bi_train, y_bi_test, y_multi_train, y_multi_test


def main():
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(
            f"Expected the AI4I 2020 CSV at '{RAW_CSV_PATH}'. "
            "Download it and place it there before running this script."
        )

    df = load_and_clean(RAW_CSV_PATH)
    X, y_binary, y_multi = prepare_features_and_targets(df)
    (
        X_bi_train,
        X_bi_test,
        y_bi_train,
        y_bi_test,
        y_multi_train,
        y_multi_test,
    ) = split(X, y_binary, y_multi)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(
        {
            "X_bi_train": X_bi_train,
            "X_bi_test": X_bi_test,
            "y_bi_train": y_bi_train,
            "y_bi_test": y_bi_test,
            "y_multi_train": y_multi_train,
            "y_multi_test": y_multi_test,
            "feature_columns": X_bi_train.columns.tolist(),
        },
        OUT_PATH,
    )
    print(f"\nSaved split data -> {OUT_PATH}")
    print(f"X_bi_train: {X_bi_train.shape}, X_bi_test: {X_bi_test.shape}")


if __name__ == "__main__":
    main()