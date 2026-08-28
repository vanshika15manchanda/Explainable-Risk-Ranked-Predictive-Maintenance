"""
compare_models.py
-----------------
Stage 2 — Multi-Label Model Comparison

This script compares candidate Stage 2 models for predicting
failure types among machines that are known to have failed.

Stage 2 predicts:
    TWF, HDF, PWF, OSF

RNF is excluded because it occurs independently of the available
process/sensor parameters and is therefore not modeled.

Candidate models:
    1. MultiOutput Logistic Regression
    2. MultiOutput Random Forest
    3. MultiOutput XGBoost
    4. MultiOutput KNN

Evaluation:
    - Micro F1
    - Macro F1
    - Hamming Loss
    - Exact Match Accuracy

The primary model-selection metric is Micro F1.

LEAKAGE PREVENTION
------------------
Stage 2 uses ONLY genuine failures from the original training split.

For every CV fold:

    Raw Stage-2 training fold
              |
       StandardScaler
              |
       Multi-output model
              |
       Validation fold
              |
          Transform only

The scaler is fitted separately inside each fold.

RNF is excluded from the modeled targets.

Depends on:
    data/processed/split_data.joblib

Produces:
    results/stage2/model_comparison.csv

Usage:
    python src/stage2/compare_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
)

import xgboost as xgb


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SPLIT_DATA_PATH = os.path.join(
    "data",
    "processed",
    "split_data.joblib"
)

RESULTS_DIR = os.path.join(
    "results",
    "stage2"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)


# ---------------------------------------------------------------------
# Failure types
# ---------------------------------------------------------------------

ALL_FAILURE_TYPES = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

FAILURE_TYPES_MODELED = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
]


# ---------------------------------------------------------------------
# Load Stage 2 data
# ---------------------------------------------------------------------

def load_stage2_data():
    """
    Load the original train/test split and keep only genuine
    failures for Stage 2.

    Returns:
        X_stage2_train
        X_stage2_test
        y_stage2_train
        y_stage2_test
    """

    data = joblib.load(
        SPLIT_DATA_PATH
    )

    X_bi_train = data["X_bi_train"]
    X_bi_test = data["X_bi_test"]

    y_bi_train = data["y_bi_train"]
    y_bi_test = data["y_bi_test"]

    y_multi_train = data["y_multi_train"]
    y_multi_test = data["y_multi_test"]

    # ---------------------------------------------------------------
    # Keep only genuine failures
    # ---------------------------------------------------------------

    train_failure_mask = (
        y_bi_train == 1
    )

    test_failure_mask = (
        y_bi_test == 1
    )

    X_stage2_train = X_bi_train[
        train_failure_mask
    ].reset_index(drop=True)

    X_stage2_test = X_bi_test[
        test_failure_mask
    ].reset_index(drop=True)

    y_stage2_train = y_multi_train[
        train_failure_mask
    ].reset_index(drop=True)

    y_stage2_test = y_multi_test[
        test_failure_mask
    ].reset_index(drop=True)

    # ---------------------------------------------------------------
    # Remove RNF
    # ---------------------------------------------------------------

    y_stage2_train = y_stage2_train[
        FAILURE_TYPES_MODELED
    ]

    y_stage2_test = y_stage2_test[
        FAILURE_TYPES_MODELED
    ]

    return (
        X_stage2_train,
        X_stage2_test,
        y_stage2_train,
        y_stage2_test,
    )


# ---------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------

def get_models(random_state=42):
    """
    Return candidate multi-output models.
    """

    models = {

        "Logistic Regression": MultiOutputClassifier(
            LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        ),

        "Random Forest": MultiOutputClassifier(
            RandomForestClassifier(
                random_state=random_state,
                n_estimators=200,
                n_jobs=-1
            )
        ),

        "XGBoost": MultiOutputClassifier(
            xgb.XGBClassifier(
                random_state=random_state,
                eval_metric="logloss"
            )
        ),

        "KNN": MultiOutputClassifier(
            KNeighborsClassifier(
                n_neighbors=5
            )
        ),
    }

    return models


# ---------------------------------------------------------------------
# Stratification target
# ---------------------------------------------------------------------

def create_stratification_target(y):
    """
    Create a single stratification target from the multi-label
    Stage 2 targets.

    This preserves the most common label combinations across folds
    where possible.

    Example:
        [1, 0, 0, 0] -> "1000"
        [0, 1, 0, 0] -> "0100"
    """

    return y.astype(str).agg(
        "".join,
        axis=1
    )


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------

def compare_models(
    X,
    y,
    n_splits=5,
    random_state=42
):
    """
    Compare candidate Stage 2 models using leakage-free
    cross-validation.
    """

    models = get_models(
        random_state=random_state
    )

    stratification_target = create_stratification_target(
        y
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for model_name, model in models.items():

        print(
            f"\n=== Evaluating {model_name} "
            f"across {n_splits} folds ==="
        )

        micro_f1_scores = []
        macro_f1_scores = []
        hamming_scores = []
        exact_match_scores = []

        for fold_num, (train_idx, val_idx) in enumerate(
            skf.split(X, stratification_target),
            start=1
        ):

            # -------------------------------------------------------
            # Split fold
            # -------------------------------------------------------

            X_train_fold = X.iloc[
                train_idx
            ]

            X_val_fold = X.iloc[
                val_idx
            ]

            y_train_fold = y.iloc[
                train_idx
            ]

            y_val_fold = y.iloc[
                val_idx
            ]

            # -------------------------------------------------------
            # Fit scaler ONLY on training fold
            # -------------------------------------------------------

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train_fold
            )

            X_val_scaled = scaler.transform(
                X_val_fold
            )

            # -------------------------------------------------------
            # Fresh model
            # -------------------------------------------------------

            fresh_model = clone(
                model
            )

            fresh_model.fit(
                X_train_scaled,
                y_train_fold
            )

            # -------------------------------------------------------
            # Prediction
            # -------------------------------------------------------

            y_pred = fresh_model.predict(
                X_val_scaled
            )

            # -------------------------------------------------------
            # Metrics
            # -------------------------------------------------------

            micro_f1 = f1_score(
                y_val_fold,
                y_pred,
                average="micro",
                zero_division=0
            )

            macro_f1 = f1_score(
                y_val_fold,
                y_pred,
                average="macro",
                zero_division=0
            )

            hamming = hamming_loss(
                y_val_fold,
                y_pred
            )

            exact_match = accuracy_score(
                y_val_fold,
                y_pred
            )

            micro_f1_scores.append(
                micro_f1
            )

            macro_f1_scores.append(
                macro_f1
            )

            hamming_scores.append(
                hamming
            )

            exact_match_scores.append(
                exact_match
            )

            print(
                f" -> Fold {fold_num} | "
                f"Micro-F1: {micro_f1:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Hamming Loss: {hamming:.4f} | "
                f"Exact Match: {exact_match:.4f}"
            )

        # -----------------------------------------------------------
        # Aggregate
        # -----------------------------------------------------------

        mean_micro_f1 = np.mean(
            micro_f1_scores
        )

        mean_macro_f1 = np.mean(
            macro_f1_scores
        )

        mean_hamming = np.mean(
            hamming_scores
        )

        mean_exact_match = np.mean(
            exact_match_scores
        )

        std_micro_f1 = np.std(
            micro_f1_scores
        )

        print(
            f"\n   Mean Micro-F1: "
            f"{mean_micro_f1:.4f} "
            f"(+/- {std_micro_f1:.4f})"
        )

        results.append({
            "Model": model_name,
            "Mean Micro-F1": mean_micro_f1,
            "Std Micro-F1": std_micro_f1,
            "Mean Macro-F1": mean_macro_f1,
            "Mean Hamming Loss": mean_hamming,
            "Mean Exact Match Accuracy": mean_exact_match,
        })

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------------
    # Rank by primary metric
    # ---------------------------------------------------------------

    results_df = results_df.sort_values(
        by="Mean Micro-F1",
        ascending=False
    ).reset_index(drop=True)

    results_df["Rank"] = (
        results_df.index + 1
    )

    return results_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(
        "===================================================="
    )

    print(
        "STAGE 2 — MULTI-LABEL MODEL COMPARISON"
    )

    print(
        "===================================================="
    )

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    (
        X_stage2_train,
        X_stage2_test,
        y_stage2_train,
        y_stage2_test,
    ) = load_stage2_data()

    print(
        f"\nStage 2 training rows: "
        f"{X_stage2_train.shape[0]}"
    )

    print(
        f"Stage 2 test rows: "
        f"{X_stage2_test.shape[0]}"
    )

    print(
        f"Failure types modeled: "
        f"{FAILURE_TYPES_MODELED}"
    )

    # ---------------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------------

    results_df = compare_models(
        X_stage2_train,
        y_stage2_train
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # ---------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------

    print(
        "\n===================================================="
    )

    print(
        "STAGE 2 MODEL COMPARISON RESULTS"
    )

    print(
        "===================================================="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    print(
        f"\nSelected model based on Mean Micro-F1:"
        f" {results_df.iloc[0]['Model']}"
    )


if __name__ == "__main__":
    main()
