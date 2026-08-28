"""
compare_models.py
-----------------
Stage 1 — Model Comparison, Rebalancing Ablation & Significance Testing

This script performs three experiments for binary machine-failure detection:

1. Compare 4 candidate Stage 1 models:
   - XGBoost
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Random Forest

   The comparison uses leakage-free 5-fold Stratified Cross-Validation.
   SMOTE and StandardScaler are fitted independently inside each fold.

2. Rebalancing ablation using XGBoost:
   - No Rebalancing
   - SMOTE
   - scale_pos_weight

3. Paired statistical significance tests comparing the F1 scores of
   each rebalancing strategy against scale_pos_weight.

LEAKAGE PREVENTION
------------------
The raw training data is split into folds BEFORE any resampling or scaling.

For every fold:

    Raw Training Fold
          |
        SMOTE
          |
    StandardScaler
          |
        Model
          |
    Validation Fold
          |
      Transform only

Thus, the validation fold never influences SMOTE or StandardScaler fitting.

Depends on:
    data/processed/split_data.joblib

Produces:
    results/stage1/model_comparison.csv
    results/stage1/rebalancing_ablation.csv
    results/stage1/significance_tests.csv

Usage:
    python src/stage1/compare_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
    "stage1"
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_split_data():
    """
    Load the binary Stage 1 training data.

    Returns:
        X_raw : Training features
        y_raw : Binary failure labels
    """

    data = joblib.load(SPLIT_DATA_PATH)

    X_raw = data["X_bi_train"].reset_index(drop=True)
    y_raw = data["y_bi_train"].reset_index(drop=True)

    return X_raw, y_raw


# ---------------------------------------------------------------------
# Experiment 1: Model Comparison
# ---------------------------------------------------------------------

def leakage_free_model_comparison(
    X_raw,
    y_raw,
    n_splits=5,
    random_state=42
):
    """
    Compare four candidate Stage 1 models using leakage-free
    Stratified K-Fold Cross-Validation.

    SMOTE and StandardScaler are fitted independently inside
    each training fold.
    """

    models = {
        "XGBoost": xgb.XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Logistic Regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            random_state=random_state
        ),
    }

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    matrix_data = {}

    for model_name, model_object in models.items():

        print(
            f"\n=== Evaluating {model_name} "
            f"across {n_splits} folds ==="
        )

        fold_accs = []
        fold_precs = []
        fold_recs = []
        fold_f1s = []

        for fold_num, (train_idx, val_idx) in enumerate(
            skf.split(X_raw, y_raw),
            start=1
        ):

            # ---------------------------------------------------------
            # Split raw data into training and validation folds
            # ---------------------------------------------------------

            X_fold_train_raw = X_raw.iloc[train_idx]
            X_fold_val_raw = X_raw.iloc[val_idx]

            y_fold_train = y_raw.iloc[train_idx]
            y_fold_val = y_raw.iloc[val_idx]

            # ---------------------------------------------------------
            # SMOTE: fit ONLY on the training fold
            # ---------------------------------------------------------

            smote_fold = SMOTE(
                random_state=random_state,
                sampling_strategy=0.5
            )

            X_res, y_res = smote_fold.fit_resample(
                X_fold_train_raw,
                y_fold_train
            )

            # ---------------------------------------------------------
            # Scaling: fit ONLY on the resampled training fold
            # ---------------------------------------------------------

            scaler_fold = StandardScaler()

            X_train_scaled = scaler_fold.fit_transform(X_res)

            X_val_scaled = scaler_fold.transform(
                X_fold_val_raw
            )

            # ---------------------------------------------------------
            # Fresh model for this fold
            # ---------------------------------------------------------

            fresh_model = clone(model_object)

            fresh_model.fit(
                X_train_scaled,
                y_res
            )

            # ---------------------------------------------------------
            # Validation prediction
            # ---------------------------------------------------------

            preds = fresh_model.predict(
                X_val_scaled
            )

            # ---------------------------------------------------------
            # Metrics
            # ---------------------------------------------------------

            acc = accuracy_score(
                y_fold_val,
                preds
            )

            prec = precision_score(
                y_fold_val,
                preds,
                zero_division=0
            )

            rec = recall_score(
                y_fold_val,
                preds,
                zero_division=0
            )

            f1 = f1_score(
                y_fold_val,
                preds,
                zero_division=0
            )

            fold_accs.append(acc)
            fold_precs.append(prec)
            fold_recs.append(rec)
            fold_f1s.append(f1)

            print(
                f" -> Fold {fold_num} | "
                f"Accuracy: {acc * 100:.1f}% | "
                f"Precision: {prec * 100:.1f}% | "
                f"Recall: {rec * 100:.1f}% | "
                f"F1: {f1 * 100:.1f}%"
            )

        # -------------------------------------------------------------
        # Store fold-level results
        # -------------------------------------------------------------

        matrix_data[
            (model_name, "Accuracy")
        ] = fold_accs

        matrix_data[
            (model_name, "Precision")
        ] = fold_precs

        matrix_data[
            (model_name, "Recall")
        ] = fold_recs

        matrix_data[
            (model_name, "F1-Score")
        ] = fold_f1s

        print(
            f"   Mean F1: "
            f"{np.mean(fold_f1s) * 100:.2f}% "
            f"(+/- {np.std(fold_f1s) * 100:.2f}%)"
        )

    # -----------------------------------------------------------------
    # Create comparison DataFrame
    # -----------------------------------------------------------------

    matrix_df = pd.DataFrame(matrix_data).T

    matrix_df.columns = [
        f"Fold {i + 1}"
        for i in range(n_splits)
    ]

    matrix_df.index.names = [
        "Model",
        "Metric"
    ]

    matrix_df["Mean CV Score"] = matrix_df.mean(axis=1)

    matrix_df["Standard Deviation"] = matrix_df.std(
        axis=1
    )

    return matrix_df


# ---------------------------------------------------------------------
# Experiment 2: Rebalancing Ablation
# ---------------------------------------------------------------------

def rebalancing_ablation(
    X_raw,
    y_raw,
    n_splits=5,
    random_state=42
):
    """
    Compare three imbalance-handling strategies using XGBoost:

        1. No Rebalancing
        2. SMOTE
        3. scale_pos_weight

    All preprocessing is performed inside each CV fold.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    ablation_results = {}

    # -----------------------------------------------------------------
    # Generic CV runner
    # -----------------------------------------------------------------

    def run_cv(method_name, fit_fn):

        accs = []
        precs = []
        recs = []
        f1s = []
        aucs = []

        for fold_num, (train_idx, val_idx) in enumerate(
            skf.split(X_raw, y_raw),
            start=1
        ):

            X_tr = X_raw.iloc[train_idx]
            X_val = X_raw.iloc[val_idx]

            y_tr = y_raw.iloc[train_idx]
            y_val = y_raw.iloc[val_idx]

            # ---------------------------------------------------------
            # Fit scaler ONLY on training fold
            # ---------------------------------------------------------

            scaler_fold = StandardScaler()

            X_tr_scaled = scaler_fold.fit_transform(X_tr)

            X_val_scaled = scaler_fold.transform(
                X_val
            )

            # ---------------------------------------------------------
            # Fit selected XGBoost strategy
            # ---------------------------------------------------------

            model = fit_fn(
                X_tr_scaled,
                y_tr
            )

            # ---------------------------------------------------------
            # Predictions
            # ---------------------------------------------------------

            preds = model.predict(
                X_val_scaled
            )

            probs = model.predict_proba(
                X_val_scaled
            )[:, 1]

            # ---------------------------------------------------------
            # Metrics
            # ---------------------------------------------------------

            accs.append(
                accuracy_score(y_val, preds)
            )

            precs.append(
                precision_score(
                    y_val,
                    preds,
                    zero_division=0
                )
            )

            recs.append(
                recall_score(
                    y_val,
                    preds,
                    zero_division=0
                )
            )

            f1s.append(
                f1_score(
                    y_val,
                    preds,
                    zero_division=0
                )
            )

            aucs.append(
                roc_auc_score(
                    y_val,
                    probs
                )
            )

        # -------------------------------------------------------------
        # Aggregate results
        # -------------------------------------------------------------

        ablation_results[method_name] = {
            "Accuracy": np.mean(accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1-Score": np.mean(f1s),
            "AUC": np.mean(aucs),
            "F1 Std": np.std(f1s),
        }

        print(
            f"{method_name}: "
            f"Mean F1 = {np.mean(f1s) * 100:.2f}% "
            f"(+/- {np.std(f1s) * 100:.2f}%)"
        )

        return f1s

    # -----------------------------------------------------------------
    # Strategy 1: No Rebalancing
    # -----------------------------------------------------------------

    def fit_none(X_tr, y_tr):

        model = xgb.XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        )

        model.fit(
            X_tr,
            y_tr
        )

        return model

    # -----------------------------------------------------------------
    # Strategy 2: SMOTE
    # -----------------------------------------------------------------

    def fit_smote(X_tr, y_tr):

        smote = SMOTE(
            random_state=random_state,
            sampling_strategy=0.5
        )

        X_res, y_res = smote.fit_resample(
            X_tr,
            y_tr
        )

        model = xgb.XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        )

        model.fit(
            X_res,
            y_res
        )

        return model

    # -----------------------------------------------------------------
    # Strategy 3: scale_pos_weight
    # -----------------------------------------------------------------

    def fit_spw(X_tr, y_tr):

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()

        scale_pos_weight = neg / pos

        model = xgb.XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

        model.fit(
            X_tr,
            y_tr
        )

        return model

    # -----------------------------------------------------------------
    # Run all three strategies
    # -----------------------------------------------------------------

    f1_folds = {}

    print("\n=== (a) No Rebalancing ===")

    f1_folds["No Rebalancing"] = run_cv(
        "No Rebalancing",
        fit_none
    )

    print("\n=== (b) SMOTE ===")

    f1_folds["SMOTE"] = run_cv(
        "SMOTE",
        fit_smote
    )

    print("\n=== (c) scale_pos_weight ===")

    f1_folds["scale_pos_weight"] = run_cv(
        "scale_pos_weight",
        fit_spw
    )

    ablation_df = pd.DataFrame(
        ablation_results
    ).T

    return ablation_df, f1_folds


# ---------------------------------------------------------------------
# Experiment 3: Statistical Significance Testing
# ---------------------------------------------------------------------

def significance_tests(
    f1_folds,
    baseline="scale_pos_weight"
):
    """
    Perform paired t-tests comparing the baseline rebalancing
    strategy against every other strategy using fold-level F1 scores.
    """

    if baseline not in f1_folds:
        raise ValueError(
            f"Baseline '{baseline}' not found in F1 results."
        )

    baseline_folds = np.asarray(
        f1_folds[baseline]
    )

    results = []

    for method_name, folds in f1_folds.items():

        if method_name == baseline:
            continue

        folds = np.asarray(folds)

        t_stat, p_value = stats.ttest_rel(
            baseline_folds,
            folds
        )

        results.append({
            "Baseline": baseline,
            "Comparison": method_name,
            "t_statistic": t_stat,
            "p_value": p_value
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    X_raw, y_raw = load_split_data()

    print(
        "--- STAGE 1: 4-MODEL COMPARISON ---"
    )

    # -----------------------------------------------------------------
    # Experiment 1
    # -----------------------------------------------------------------

    matrix_df = leakage_free_model_comparison(
        X_raw,
        y_raw
    )

    model_comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    matrix_df.to_csv(
        model_comparison_path
    )

    print("\nModel Comparison Results:")
    print(matrix_df)

    # -----------------------------------------------------------------
    # Experiment 2
    # -----------------------------------------------------------------

    print(
        "\n--- STAGE 1: REBALANCING ABLATION ---"
    )

    ablation_df, f1_folds = rebalancing_ablation(
        X_raw,
        y_raw
    )

    ablation_path = os.path.join(
        RESULTS_DIR,
        "rebalancing_ablation.csv"
    )

    ablation_df.to_csv(
        ablation_path
    )

    print("\nRebalancing Ablation Results:")
    print(ablation_df)

    # -----------------------------------------------------------------
    # Experiment 3
    # -----------------------------------------------------------------

    print(
        "\n--- STAGE 1: SIGNIFICANCE TESTING ---"
    )

    sig_df = significance_tests(
        f1_folds,
        baseline="scale_pos_weight"
    )

    significance_path = os.path.join(
        RESULTS_DIR,
        "significance_tests.csv"
    )

    sig_df.to_csv(
        significance_path,
        index=False
    )

    print("\nSignificance Test Results:")
    print(sig_df)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print(
        f"\nResults saved to: {RESULTS_DIR}/"
    )


if __name__ == "__main__":
    main()