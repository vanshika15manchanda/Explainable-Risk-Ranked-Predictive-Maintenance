"""
cascade_pipeline.py
--------------------
Final stage of the pipeline: wires Stage 1 (binary failure detection)
and Stage 2 (failure-type classification) into one end-to-end
cascade, evaluates it HONESTLY (including cascade contamination —
see below), computes SHAP-driven priority scores, and produces a
ranked maintenance schedule.

Both stages are now XGBoost, so both get SHAP's TreeExplainer
(exact and fast for tree models — no approximation, and no need for
LinearExplainer now that Stage 2 isn't Logistic Regression).

CASCADE CONTAMINATION — why this file evaluates two different ways
--------------------------------------------------------------------
The "naive" way to evaluate a cascade is: filter Stage 2's test set
down to ROWS STAGE 1 CORRECTLY FLAGGED, then score Stage 2 only on
those. That's what compare_models.py / train_final.py for Stage 2
both do, and it's the right way to CHOOSE a Stage 2 model.

But it's not what happens in deployment. In deployment, Stage 2 gets
EVERY row Stage 1 flags — including Stage 1's false positives. Stage
2 has no "not actually a failure" option; it always outputs some
failure-type prediction. So every false positive becomes a
confident-looking, fabricated diagnosis, invisible to the standard
metrics because those metrics are only ever computed on genuine
failures.

This script reports both numbers so the gap itself is visible:
    - "Stage 2 isolated" accuracy  = the optimistic number (true
      failures only, same as train_final.py reports)
    - "True cascade" accuracy      = Stage 2's accuracy across
      EVERY row Stage 1 flagged, false positives included, scored
      against ground truth (false positives are automatically wrong)

Depends on:
    data/processed/split_data.joblib
    models/stage1_xgb_model.joblib
    models/stage1_scaler.joblib
    models/stage2_xgb_model.joblib
    models/pipeline_metadata.joblib

Produces:
    results/cascade_evaluation.csv       — both evaluation modes
    results/priority_ranking.csv         — ranked maintenance schedule
    models/shap_background.joblib        — background sets for the app

Usage:
    python src/cascade_pipeline.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)

SPLIT_DATA_PATH = os.path.join("data", "processed", "split_data.joblib")
MODELS_DIR = "models"
RESULTS_DIR = "results"

STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "stage1_xgb_model.joblib")
STAGE1_SCALER_PATH = os.path.join(MODELS_DIR, "stage1_scaler.joblib")
STAGE2_MODEL_PATH = os.path.join(MODELS_DIR, "stage2_xgb_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "pipeline_metadata.joblib")

# Priority score weights — Stage 1 urgency, Stage 2 confidence, SHAP magnitude
W1_STAGE1_PROB = 0.4
W2_STAGE2_CONF = 0.3
W3_SHAP_MAGNITUDE = 0.3


def load_artifacts():
    data = joblib.load(SPLIT_DATA_PATH)
    for path, label in [
        (STAGE1_MODEL_PATH, "Stage 1 model"),
        (STAGE1_SCALER_PATH, "Stage 1 scaler"),
        (STAGE2_MODEL_PATH, "Stage 2 model"),
        (METADATA_PATH, "pipeline metadata"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found at '{path}' — run the training scripts first.")

    stage1_model = joblib.load(STAGE1_MODEL_PATH)
    stage1_scaler = joblib.load(STAGE1_SCALER_PATH)
    stage2_model = joblib.load(STAGE2_MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return data, stage1_model, stage1_scaler, stage2_model, metadata


def run_stage1(stage1_model, stage1_scaler, X_test):
    X_test_scaled = stage1_scaler.transform(X_test)
    stage1_probs = stage1_model.predict_proba(X_test_scaled)[:, 1]
    stage1_preds = stage1_model.predict(X_test_scaled)
    return X_test_scaled, stage1_probs, stage1_preds


def evaluate_stage2_isolated(stage2_model, X_test_scaled, y_bi_test, y_multi_test, failure_types):
    """Optimistic number: Stage 2 scored only on genuine failures — matches train_final.py's report."""
    true_failure_mask = y_bi_test == 1
    X_true = X_test_scaled[true_failure_mask.values]
    y_true = y_multi_test.loc[true_failure_mask][failure_types]

    y_pred = stage2_model.predict(X_true)
    return {
        "n_rows": int(true_failure_mask.sum()),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_true_cascade(stage2_model, X_test_scaled, stage1_preds, y_bi_test, y_multi_test, failure_types):
    """Honest number: Stage 2 scored on EVERY row Stage 1 flagged, false positives included.
    A false positive can never be "correct" here since ground truth has no failure type."""
    flagged_mask = stage1_preds == 1
    n_flagged = int(flagged_mask.sum())
    if n_flagged == 0:
        return {"n_flagged": 0, "n_false_positives": 0, "contamination_rate": 0.0, "true_cascade_exact_match": None}

    X_flagged = X_test_scaled[flagged_mask]
    y_pred_flagged = stage2_model.predict(X_flagged)

    true_bi = y_bi_test.values[flagged_mask]
    false_positive_mask = true_bi == 0
    n_false_positives = int(false_positive_mask.sum())

    # Ground truth for flagged rows: real failure types where the row IS a real failure,
    # all-zero (i.e. automatically wrong) where it's a false positive
    y_true_flagged = np.zeros((n_flagged, len(failure_types)), dtype=int)
    real_failure_positions = np.where(true_bi == 1)[0]
    if len(real_failure_positions) > 0:
        y_true_flagged[real_failure_positions] = (
            y_multi_test.loc[y_bi_test[flagged_mask].index[real_failure_positions]][failure_types].values
        )

    exact_match = accuracy_score(y_true_flagged, y_pred_flagged)

    return {
        "n_flagged": n_flagged,
        "n_false_positives": n_false_positives,
        "contamination_rate": n_false_positives / n_flagged,
        "true_cascade_exact_match": exact_match,
    }


def compute_shap_priority(
    stage1_model, stage2_model, X_test, X_test_scaled, stage1_probs, stage1_preds, stage2_model_ref, failure_types
):
    """SHAP magnitude (TreeExplainer, exact for both stages now they're both XGBoost)
    combined with Stage 1 probability and Stage 2 confidence into a priority score."""
    flagged_idx = np.where(stage1_preds == 1)[0]
    if len(flagged_idx) == 0:
        return pd.DataFrame(), None, None

    X_flagged_scaled = X_test_scaled[flagged_idx]

    # Stage 1 SHAP — magnitude of the evidence that pushed toward "failure"
    stage1_explainer = shap.TreeExplainer(stage1_model)
    stage1_shap_values = stage1_explainer.shap_values(X_flagged_scaled)
    stage1_shap_magnitude = np.abs(stage1_shap_values).sum(axis=1)

    # Stage 2 SHAP — magnitude of evidence behind the predicted failure type(s),
    # summed across the MultiOutputClassifier's per-label estimators
    stage2_probs_per_label = np.stack(
        [est.predict_proba(X_flagged_scaled)[:, 1] for est in stage2_model.estimators_], axis=1
    )
    stage2_confidence = stage2_probs_per_label.max(axis=1)

    stage2_shap_magnitude = np.zeros(len(flagged_idx))
    for est in stage2_model.estimators_:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_flagged_scaled)
        sv = sv[1] if isinstance(sv, list) else sv
        stage2_shap_magnitude += np.abs(sv).sum(axis=1)

    combined_shap_magnitude = stage1_shap_magnitude + stage2_shap_magnitude

    priority_frame = pd.DataFrame(
        {
            "row_index": X_test.index[flagged_idx],
            "stage1_failure_probability": stage1_probs[flagged_idx],
            "stage2_confidence": stage2_confidence,
            "shap_magnitude": combined_shap_magnitude,
            "predicted_failure_types": [
                ", ".join([ft for ft, p in zip(failure_types, row) if p == 1]) or "none"
                for row in stage2_model.predict(X_flagged_scaled)
            ],
        }
    )

    # Normalize each component to [0, 1] within this batch before combining
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(
        priority_frame[["stage1_failure_probability", "stage2_confidence", "shap_magnitude"]]
    )
    priority_frame["priority_score"] = (
        W1_STAGE1_PROB * normalized[:, 0] + W2_STAGE2_CONF * normalized[:, 1] + W3_SHAP_MAGNITUDE * normalized[:, 2]
    )

    priority_frame = priority_frame.sort_values("priority_score", ascending=False).reset_index(drop=True)
    priority_frame["priority_rank"] = priority_frame.index + 1

    shap_background = {
        "stage1_explainer_background": X_flagged_scaled[: min(100, len(X_flagged_scaled))],
    }

    return priority_frame, shap_background, scaler


def main():
    print("=" * 70)
    print("CASCADE PIPELINE — Stage 1 -> Stage 2 -> Priority Ranking")
    print("=" * 70)

    data, stage1_model, stage1_scaler, stage2_model, metadata = load_artifacts()
    failure_types = metadata["failure_type_names"]

    X_test = data["X_bi_test"]
    y_bi_test = data["y_bi_test"]
    y_multi_test = data["y_multi_test"]

    print("\nRunning Stage 1 on the full test set...")
    X_test_scaled, stage1_probs, stage1_preds = run_stage1(stage1_model, stage1_scaler, X_test)

    stage1_precision = precision_score(y_bi_test, stage1_preds, zero_division=0)
    stage1_recall = recall_score(y_bi_test, stage1_preds, zero_division=0)
    print(f"Stage 1 — Precision: {stage1_precision:.4f} | Recall: {stage1_recall:.4f}")

    print("\n--- Stage 2 evaluation: isolated vs. true cascade ---")
    isolated_results = evaluate_stage2_isolated(stage2_model, X_test_scaled, y_bi_test, y_multi_test, failure_types)
    cascade_results = evaluate_true_cascade(
        stage2_model, X_test_scaled, stage1_preds, y_bi_test, y_multi_test, failure_types
    )

    print(f"\nStage 2 isolated (true failures only, n={isolated_results['n_rows']}):")
    print(f"  Micro-F1: {isolated_results['micro_f1']:.4f} | Exact Match: {isolated_results['exact_match_accuracy']:.4f}")

    print(f"\nTrue cascade (all {cascade_results['n_flagged']} flagged rows, {cascade_results['n_false_positives']} false positives):")
    print(f"  Contamination rate: {cascade_results['contamination_rate']*100:.1f}%")
    print(f"  True cascade exact match: {cascade_results['true_cascade_exact_match']:.4f}")

    print("\nComputing SHAP-driven priority scores for flagged machines...")
    priority_frame, shap_background, _ = compute_shap_priority(
        stage1_model, stage2_model, X_test, X_test_scaled, stage1_probs, stage1_preds, stage2_model, failure_types
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_summary = pd.DataFrame(
        [
            {"Metric": "Stage 1 Precision", "Value": stage1_precision},
            {"Metric": "Stage 1 Recall", "Value": stage1_recall},
            {"Metric": "Stage 2 Isolated Micro-F1", "Value": isolated_results["micro_f1"]},
            {"Metric": "Stage 2 Isolated Exact Match", "Value": isolated_results["exact_match_accuracy"]},
            {"Metric": "Cascade Contamination Rate", "Value": cascade_results["contamination_rate"]},
            {"Metric": "True Cascade Exact Match", "Value": cascade_results["true_cascade_exact_match"]},
        ]
    )
    eval_summary.to_csv(os.path.join(RESULTS_DIR, "cascade_evaluation.csv"), index=False)
    priority_frame.to_csv(os.path.join(RESULTS_DIR, "priority_ranking.csv"), index=False)

    if shap_background is not None:
        joblib.dump(shap_background, os.path.join(MODELS_DIR, "shap_background.joblib"))

    print(f"\nSaved: {RESULTS_DIR}/cascade_evaluation.csv")
    print(f"Saved: {RESULTS_DIR}/priority_ranking.csv  (top of the maintenance schedule)")
    print("\nTop 5 machines by priority:")
    print(priority_frame.head(5).to_string(index=False))


if __name__ == "__main__":
    main()