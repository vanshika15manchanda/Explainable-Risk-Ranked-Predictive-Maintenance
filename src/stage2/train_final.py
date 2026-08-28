"""
train_final.py
--------------
STEP 8b of the pipeline: hyperparameter-tune and train the final
Stage 2 model.

Stage 2 task:
    Predict WHICH failure type(s) occurred among machines that
    are already known to have failed.

Modeled failure types:
    TWF, HDF, PWF, OSF

RNF is excluded because it is considered unpredictable from the
available process/sensor features.

MODEL SELECTION
---------------
Stage 2 model comparison showed:

    XGBoost              -> Micro-F1 = 0.9380  [SELECTED]
    Random Forest        -> Micro-F1 = 0.9300
    Logistic Regression  -> Micro-F1 = 0.9072
    KNN                   -> Micro-F1 = 0.8277

Therefore, XGBoost is selected for hyperparameter tuning.

SCALER
------
Stage 2 reuses the exact StandardScaler fitted during Stage 1.

The scaler is loaded from:
    models/scaler_final.joblib

It is NEVER refitted here.

DATA
----
Stage 2 uses only genuine failures from the original train/test
split. No SMOTE or synthetic samples are used.

PRODUCES
--------
models/stage2_xgb_model.joblib
models/pipeline_metadata.joblib
results/stage2/final_model_results.csv

Usage:
    python src/stage2/train_final.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    classification_report,
)


# ================================================================
# PATHS
# ================================================================

SPLIT_DATA_PATH = os.path.join(
    "data",
    "processed",
    "split_data.joblib"
)

MODELS_DIR = "models"

SCALER_PATH = os.path.join(
    MODELS_DIR,
    "stage1_scaler.joblib"
)

STAGE2_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "stage2_xgb_model.joblib"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "pipeline_metadata.joblib"
)

RESULTS_DIR = os.path.join(
    "results",
    "stage2"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "final_model_results.csv"
)


# ================================================================
# FAILURE TYPES
# ================================================================

FAILURE_TYPES_MODELED = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
]


# ================================================================
# LOAD STAGE 2 DATA
# ================================================================

def prepare_stage2_data(data):
    """
    Keep only genuine failures from the original train/test split.

    Stage 2 does not use:
        - normal machines
        - SMOTE samples
        - synthetic failures

    RNF is removed from the target labels.
    """

    X_bi_train = data["X_bi_train"]
    X_bi_test = data["X_bi_test"]

    y_bi_train = data["y_bi_train"]
    y_bi_test = data["y_bi_test"]

    y_multi_train = data["y_multi_train"]
    y_multi_test = data["y_multi_test"]

    # ------------------------------------------------------------
    # Keep only actual failures
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Remove RNF
    # ------------------------------------------------------------

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


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("STEP 8b — FINAL STAGE 2 MODEL TRAINING")
    print("=" * 70)

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("\nLoading split data...")

    data = joblib.load(
        SPLIT_DATA_PATH
    )

    (
        X_stage2_train,
        X_stage2_test,
        y_stage2_train,
        y_stage2_test,
    ) = prepare_stage2_data(data)

    print(
        f"Stage 2 training rows: "
        f"{X_stage2_train.shape[0]}"
    )

    print(
        f"Stage 2 test rows: "
        f"{X_stage2_test.shape[0]}"
    )

    print(
        f"Features: {X_stage2_train.shape[1]}"
    )

    print(
        f"Failure types: "
        f"{FAILURE_TYPES_MODELED}"
    )

    # ============================================================
    # LOAD STAGE 1 SCALER
    # ============================================================

    print("\nLoading Stage 1 scaler...")

    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            f"\n'{SCALER_PATH}' not found.\n"
            "Run src/stage1/train_final.py first."
        )

    scaler_final = joblib.load(
        SCALER_PATH
    )

    # IMPORTANT:
    # The scaler is already fitted.
    # Do NOT fit it again.

    X_stage2_train_scaled = (
        scaler_final.transform(
            X_stage2_train
        )
    )

    X_stage2_test_scaled = (
        scaler_final.transform(
            X_stage2_test
        )
    )

    print(
        "Stage 1 scaler successfully reused."
    )

    # ============================================================
    # BASE STAGE 2 MODEL
    # ============================================================

    print(
        "\nSelected model: "
        "MultiOutput XGBoost"
    )

    base_xgb = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    multi_output_xgb = MultiOutputClassifier(
        base_xgb,
        n_jobs=-1
    )

    # ============================================================
    # HYPERPARAMETER GRID
    # ============================================================

    param_grid = {

        "estimator__n_estimators": [
            100,
            200,
            300
        ],

        "estimator__learning_rate": [
            0.01,
            0.05,
            0.1
        ],

        "estimator__max_depth": [
            3,
            5,
            7
        ],

        "estimator__subsample": [
            0.8,
            1.0
        ],

        "estimator__colsample_bytree": [
            0.8,
            1.0
        ],
    }

    # ============================================================
    # GRID SEARCH
    # ============================================================

    print(
        "\nStarting GridSearchCV..."
    )

    print(
        "Scoring metric: F1-micro"
    )

    grid_search = GridSearchCV(
        estimator=multi_output_xgb,
        param_grid=param_grid,
        scoring="f1_micro",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(
        X_stage2_train_scaled,
        y_stage2_train
    )

    # ============================================================
    # BEST PARAMETERS
    # ============================================================

    best_params = (
        grid_search.best_params_
    )

    best_cv_f1 = (
        grid_search.best_score_
    )

    print("\n" + "=" * 70)
    print("BEST STAGE 2 PARAMETERS")
    print("=" * 70)

    for parameter, value in best_params.items():

        print(
            f"{parameter}: {value}"
        )

    print(
        f"\nBest CV Micro-F1: "
        f"{best_cv_f1:.4f}"
    )

    # ============================================================
    # TRAIN FINAL MODEL
    # ============================================================

    print(
        "\nTraining final Stage 2 XGBoost..."
    )

    final_xgb = xgb.XGBClassifier(
        n_estimators=best_params[
            "estimator__n_estimators"
        ],

        learning_rate=best_params[
            "estimator__learning_rate"
        ],

        max_depth=best_params[
            "estimator__max_depth"
        ],

        subsample=best_params[
            "estimator__subsample"
        ],

        colsample_bytree=best_params[
            "estimator__colsample_bytree"
        ],

        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    final_stage2_model = MultiOutputClassifier(
        final_xgb,
        n_jobs=-1
    )

    final_stage2_model.fit(
        X_stage2_train_scaled,
        y_stage2_train
    )

    # ============================================================
    # TEST SET PREDICTION
    # ============================================================

    print(
        "\nEvaluating final model on untouched test set..."
    )

    y_pred = final_stage2_model.predict(
        X_stage2_test_scaled
    )

    # ============================================================
    # METRICS
    # ============================================================

    micro_f1 = f1_score(
        y_stage2_test,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_stage2_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_stage2_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    hamming = hamming_loss(
        y_stage2_test,
        y_pred
    )

    exact_match = accuracy_score(
        y_stage2_test,
        y_pred
    )

    # ============================================================
    # RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL STAGE 2 TEST RESULTS")
    print("=" * 70)

    print(
        f"Micro F1              : "
        f"{micro_f1:.4f}"
    )

    print(
        f"Macro F1              : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1           : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Hamming Loss          : "
        f"{hamming:.4f}"
    )

    print(
        f"Exact Match Accuracy  : "
        f"{exact_match:.4f}"
    )

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_stage2_test,
            y_pred,
            target_names=FAILURE_TYPES_MODELED,
            zero_division=0
        )
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(
        final_stage2_model,
        STAGE2_MODEL_PATH
    )

    # ============================================================
    # SAVE METADATA
    # ============================================================

    pipeline_metadata = {
        "feature_columns": data[
            "feature_columns"
        ],

        "failure_type_names": (
            FAILURE_TYPES_MODELED
        ),

        "stage2_model": (
            "MultiOutputClassifier("
            "XGBClassifier)"
        ),

        "best_parameters": best_params,

        "best_cv_micro_f1": (
            best_cv_f1
        ),

        "test_micro_f1": (
            micro_f1
        ),

        "test_macro_f1": (
            macro_f1
        ),

        "test_weighted_f1": (
            weighted_f1
        ),

        "test_hamming_loss": (
            hamming
        ),

        "test_exact_match_accuracy": (
            exact_match
        ),
    }

    joblib.dump(
        pipeline_metadata,
        METADATA_PATH
    )

    # ============================================================
    # SAVE RESULTS CSV
    # ============================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_df = pd.DataFrame([
        {
            "Model": (
                "MultiOutput XGBoost"
            ),

            "Task": (
                "Stage 2 Multi-Label "
                "Failure-Type Classification"
            ),

            "Failure Types": (
                ", ".join(
                    FAILURE_TYPES_MODELED
                )
            ),

            "Best CV Micro-F1": (
                best_cv_f1
            ),

            "Test Micro-F1": (
                micro_f1
            ),

            "Test Macro-F1": (
                macro_f1
            ),

            "Test Weighted-F1": (
                weighted_f1
            ),

            "Test Hamming Loss": (
                hamming
            ),

            "Test Exact Match Accuracy": (
                exact_match
            ),

            "Best Parameters": (
                str(best_params)
            ),
        }
    ])

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nFinal Stage 2 model:"
        f"\n{STAGE2_MODEL_PATH}"
    )

    print(
        f"\nPipeline metadata:"
        f"\n{METADATA_PATH}"
    )

    print(
        f"\nFinal results:"
        f"\n{RESULTS_PATH}"
    )

    print("\nStage 2 training completed successfully.")


if __name__ == "__main__":
    main()
