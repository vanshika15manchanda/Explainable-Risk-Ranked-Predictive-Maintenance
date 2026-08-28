"""
train_final.py
--------------
Stage 1 — Final XGBoost Training & Hyperparameter Tuning

Model-selection decisions established in compare_models.py:
    - Imbalance strategy: scale_pos_weight
    - SMOTE: NOT used for the final model

Workflow:
    Raw training data
          |
    StandardScaler
          |
    GridSearchCV
          |
    XGBoost + scale_pos_weight
          |
    Best hyperparameters
          |
    Final Stage 1 model
          |
    Saved model + scaler

IMPORTANT:
    The scaler is fitted ONLY on the raw training data.
    The test set is never used during hyperparameter tuning.

Depends on:
    data/processed/split_data.joblib

Produces:
    models/stage1_xgb_model.joblib
    models/stage1_scaler.joblib
    results/stage1/final_model_results.csv

Usage:
    python src/stage1/train_final.py
"""

import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SPLIT_DATA_PATH = os.path.join(
    "data",
    "processed",
    "split_data.joblib"
)

MODELS_DIR = "models"

RESULTS_DIR = os.path.join(
    "results",
    "stage1"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "stage1_xgb_model.joblib"
)

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "stage1_scaler.joblib"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "final_model_results.csv"
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_split_data():
    """
    Load the binary Stage 1 train/test split.
    """

    data = joblib.load(
        SPLIT_DATA_PATH
    )

    X_train = data["X_bi_train"]
    X_test = data["X_bi_test"]

    y_train = data["y_bi_train"]
    y_test = data["y_bi_test"]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------

def main():

    print(
        "--- STAGE 1: FINAL XGBOOST TRAINING ---"
    )

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_split_data()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # -----------------------------------------------------------------
    # Scale features
    # -----------------------------------------------------------------
    #
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # Test data is transformed using the same scaler.
    #
    # No SMOTE is used here because the selected imbalance strategy
    # from the ablation study is scale_pos_weight.
    # -----------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -----------------------------------------------------------------
    # Calculate class imbalance
    # -----------------------------------------------------------------

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    base_scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nNegative samples: {negative_count}"
    )

    print(
        f"Positive samples: {positive_count}"
    )

    print(
        f"Base scale_pos_weight: "
        f"{base_scale_pos_weight:.2f}"
    )

    # -----------------------------------------------------------------
    # Hyperparameter grid
    # -----------------------------------------------------------------

    param_grid = {
        "n_estimators": [
            100,
            200,
            300
        ],

        "learning_rate": [
            0.01,
            0.1,
            0.2
        ],

        "max_depth": [
            3,
            5,
            7
        ],

        "scale_pos_weight": [
            1,
            base_scale_pos_weight * 0.5,
            base_scale_pos_weight,
            base_scale_pos_weight * 1.5
        ],
    }

    # -----------------------------------------------------------------
    # Base XGBoost model
    # -----------------------------------------------------------------

    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    # -----------------------------------------------------------------
    # GridSearchCV
    # -----------------------------------------------------------------

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    print(
        "\nStarting GridSearchCV..."
    )

    grid_search.fit(
        X_train_scaled,
        y_train
    )

    # -----------------------------------------------------------------
    # Best parameters
    # -----------------------------------------------------------------

    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_

    best_cv_f1 = grid_search.best_score_

    print(
        "\n--- BEST MODEL ---"
    )

    print(
        "Best parameters:"
    )

    for parameter, value in best_params.items():
        print(
            f"  {parameter}: {value}"
        )

    print(
        f"\nBest CV F1-score: "
        f"{best_cv_f1:.4f}"
    )

    # -----------------------------------------------------------------
    # Test-set evaluation
    # -----------------------------------------------------------------

    y_pred = best_model.predict(
        X_test_scaled
    )

    y_prob = best_model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    # -----------------------------------------------------------------
    # Print test results
    # -----------------------------------------------------------------

    print(
        "\n--- FINAL STAGE 1 TEST RESULTS ---"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # -----------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    # -----------------------------------------------------------------
    # Save scaler
    # -----------------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # -----------------------------------------------------------------
    # Save final results
    # -----------------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df = pd.DataFrame([
        {
            "Model": "XGBoost",
            "Task": "Stage 1 Binary Failure Detection",
            "Best CV F1": best_cv_f1,
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1": f1,
            "Test ROC-AUC": auc,
            "Best Parameters": str(best_params)
        }
    ])

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # -----------------------------------------------------------------
    # Final confirmation
    # -----------------------------------------------------------------

    print(
        "\n--- FILES SAVED ---"
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Scaler: {SCALER_PATH}"
    )

    print(
        f"Results: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
