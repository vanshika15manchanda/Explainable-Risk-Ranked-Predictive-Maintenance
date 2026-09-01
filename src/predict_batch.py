"""
predict_batch.py
-----------------
Takes a CSV of RAW machine readings (same columns as the original
AI4I 2020 dataset, minus the target columns) and runs it through the
full cascade: Stage 1 -> Stage 2 -> SHAP priority scoring -> ranked
CSV output. This is the "new data in, ranked schedule out" entry
point — cascade_pipeline.py evaluates against the KNOWN test split
(it needs ground truth to report accuracy); this script is for data
where you don't have ground truth yet, which is the real deployment
case.

Expected input CSV columns (raw, unencoded):
    Type, Air temperature [K], Process temperature [K],
    Rotational speed [rpm], Torque [Nm], Tool wear [min]

('Type' should be one of L / M / H — the script one-hot encodes it
the same way data_prep.py did.)

Usage:
    python src/predict_batch.py --input data/new_readings.csv --output results/new_ranking.csv
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import MinMaxScaler

MACHINE_ID_COLUMN = "Machine ID"

MODELS_DIR = "models"
STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "stage1_xgb_model.joblib")
STAGE1_SCALER_PATH = os.path.join(MODELS_DIR, "stage1_scaler.joblib")
STAGE2_MODEL_PATH = os.path.join(MODELS_DIR, "stage2_xgb_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "pipeline_metadata.joblib")

W1_STAGE1_PROB = 0.4
W2_STAGE2_CONF = 0.3
W3_SHAP_MAGNITUDE = 0.3


def load_artifacts():
    for path, label in [
        (STAGE1_MODEL_PATH, "Stage 1 model"),
        (STAGE1_SCALER_PATH, "Stage 1 scaler"),
        (STAGE2_MODEL_PATH, "Stage 2 model"),
        (METADATA_PATH, "pipeline metadata"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found at '{path}' — run the training scripts first.")

    return (
        joblib.load(STAGE1_MODEL_PATH),
        joblib.load(STAGE1_SCALER_PATH),
        joblib.load(STAGE2_MODEL_PATH),
        joblib.load(METADATA_PATH),
    )


def encode_raw(df_raw: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Same one-hot encoding data_prep.py used on 'Type', then align
    columns exactly to what the models were trained on."""
    df_encoded = pd.get_dummies(df_raw, columns=["Type"], drop_first=True, dtype=int)

    # Add any dummy columns the model expects but this batch didn't produce
    # (e.g. a batch with only 'L' machines won't create Type_M / Type_H)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[feature_columns]


# def score_batch(X_raw, stage1_model, stage1_scaler, stage2_model, failure_types):
#     X_scaled = stage1_scaler.transform(X_raw)

#     stage1_probs = stage1_model.predict_proba(X_scaled)[:, 1]
#     stage1_preds = stage1_model.predict(X_scaled)

#     flagged_idx = np.where(stage1_preds == 1)[0]

#     result = X_raw.copy()
#     result["stage1_failure_probability"] = stage1_probs
#     result["stage1_flagged"] = stage1_preds

#     if len(flagged_idx) == 0:
#         result["predicted_failure_types"] = "n/a (not flagged)"
#         result["stage2_confidence"] = np.nan
#         result["shap_magnitude"] = np.nan
#         result["priority_score"] = np.nan
#         result["priority_rank"] = np.nan
#         return result.sort_values("stage1_failure_probability", ascending=False)

#     X_flagged_scaled = X_scaled[flagged_idx]

#     stage2_preds = stage2_model.predict(X_flagged_scaled)
#     stage2_probs_per_label = np.stack(
#         [est.predict_proba(X_flagged_scaled)[:, 1] for est in stage2_model.estimators_], axis=1
#     )
#     stage2_confidence = stage2_probs_per_label.max(axis=1)

#     stage1_explainer = shap.TreeExplainer(stage1_model)
#     stage1_shap_magnitude = np.abs(stage1_explainer.shap_values(X_flagged_scaled)).sum(axis=1)

#     stage2_shap_magnitude = np.zeros(len(flagged_idx))
#     for est in stage2_model.estimators_:
#         explainer = shap.TreeExplainer(est)
#         sv = explainer.shap_values(X_flagged_scaled)
#         sv = sv[1] if isinstance(sv, list) else sv
#         stage2_shap_magnitude += np.abs(sv).sum(axis=1)

#     combined_shap_magnitude = stage1_shap_magnitude + stage2_shap_magnitude

#     predicted_types = [
#         ", ".join([ft for ft, p in zip(failure_types, row) if p == 1]) or "none"
#         for row in stage2_preds
#     ]

#     scaler = MinMaxScaler()
#     normalized = scaler.fit_transform(
#         np.column_stack([stage1_probs[flagged_idx], stage2_confidence, combined_shap_magnitude])
#     )
#     priority_scores = (
#         W1_STAGE1_PROB * normalized[:, 0] + W2_STAGE2_CONF * normalized[:, 1] + W3_SHAP_MAGNITUDE * normalized[:, 2]
#     )

#     result["predicted_failure_types"] = "n/a (not flagged)"
#     result["stage2_confidence"] = np.nan
#     result["shap_magnitude"] = np.nan
#     result["priority_score"] = np.nan

#     result.iloc[flagged_idx, result.columns.get_loc("predicted_failure_types")] = predicted_types
#     result.iloc[flagged_idx, result.columns.get_loc("stage2_confidence")] = stage2_confidence
#     result.iloc[flagged_idx, result.columns.get_loc("shap_magnitude")] = combined_shap_magnitude
#     result.iloc[flagged_idx, result.columns.get_loc("priority_score")] = priority_scores

#     result = result.sort_values(["stage1_flagged", "priority_score"], ascending=[False, False])
#     result["priority_rank"] = range(1, len(result) + 1)

#     # Keep the ORIGINAL row position (not reset) so callers like main() can
#     # still map extra columns (e.g. Machine ID) back by original row order
#     return result


def score_batch(X_raw, stage1_model, stage1_scaler, stage2_model, failure_types):
    X_scaled = stage1_scaler.transform(X_raw)

    stage1_probs = stage1_model.predict_proba(X_scaled)[:, 1]
    stage1_preds = stage1_model.predict(X_scaled)

    flagged_idx = np.where(stage1_preds == 1)[0]

    result = X_raw.copy()
    result["stage1_failure_probability"] = stage1_probs
    result["stage1_flagged"] = stage1_preds
    result["top_features"] = [[] for _ in range(len(result))]

    if len(flagged_idx) == 0:
        result["predicted_failure_types"] = "n/a (not flagged)"
        result["stage2_confidence"] = np.nan
        result["shap_magnitude"] = np.nan
        result["priority_score"] = np.nan
        result["priority_rank"] = np.nan
        return result.sort_values("stage1_failure_probability", ascending=False)

    X_flagged_scaled = X_scaled[flagged_idx]

    stage2_preds = stage2_model.predict(X_flagged_scaled)
    stage2_probs_per_label = np.stack(
        [est.predict_proba(X_flagged_scaled)[:, 1] for est in stage2_model.estimators_], axis=1
    )
    stage2_confidence = stage2_probs_per_label.max(axis=1)

    stage1_explainer = shap.TreeExplainer(stage1_model)
    stage1_shap_values = stage1_explainer.shap_values(X_flagged_scaled)
    stage1_shap_magnitude = np.abs(stage1_shap_values).sum(axis=1)

    stage2_shap_magnitude = np.zeros(len(flagged_idx))
    for est in stage2_model.estimators_:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_flagged_scaled)
        sv = sv[1] if isinstance(sv, list) else sv
        stage2_shap_magnitude += np.abs(sv).sum(axis=1)

    combined_shap_magnitude = stage1_shap_magnitude + stage2_shap_magnitude

    predicted_types = [
        ", ".join([ft for ft, p in zip(failure_types, row) if p == 1]) or "none"
        for row in stage2_preds
    ]

    # Per-feature SHAP breakdown (top 4 by magnitude) for each flagged machine.
    # Uses Stage 1's signed SHAP values -- positive pushes toward failure, negative away.
    # Display-only: does not change any model behavior.
    top_features_list = []
    for i, row_shap in enumerate(stage1_shap_values):
        pairs = list(zip(X_raw.columns, row_shap))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top4 = pairs[:4]
        original_row = X_raw.iloc[flagged_idx[i]]
        top_features_list.append([
            {"feature": name, "value": round(float(original_row[name]), 2), "impact": round(float(val), 4)}
            for name, val in top4
        ])

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(
        np.column_stack([stage1_probs[flagged_idx], stage2_confidence, combined_shap_magnitude])
    )
    priority_scores = (
        W1_STAGE1_PROB * normalized[:, 0] + W2_STAGE2_CONF * normalized[:, 1] + W3_SHAP_MAGNITUDE * normalized[:, 2]
    )

    result["predicted_failure_types"] = "n/a (not flagged)"
    result["stage2_confidence"] = np.nan
    result["shap_magnitude"] = np.nan
    result["priority_score"] = np.nan
    result["top_features"] = result["top_features"].astype(object)

    result.iloc[flagged_idx, result.columns.get_loc("predicted_failure_types")] = predicted_types
    result.iloc[flagged_idx, result.columns.get_loc("stage2_confidence")] = stage2_confidence
    result.iloc[flagged_idx, result.columns.get_loc("shap_magnitude")] = combined_shap_magnitude
    result.iloc[flagged_idx, result.columns.get_loc("priority_score")] = priority_scores
    for i, idx in enumerate(flagged_idx):
        result.iat[idx, result.columns.get_loc("top_features")] = top_features_list[i]

    result = result.sort_values(["stage1_flagged", "priority_score"], ascending=[False, False])
    result["priority_rank"] = range(1, len(result) + 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Score a batch of raw machine readings through the cascade.")
    parser.add_argument("--input", required=True, help="Path to input CSV of raw readings.")
    parser.add_argument("--output", default=os.path.join("results", "new_ranking.csv"), help="Path to write ranked output CSV.")
    args = parser.parse_args()

    stage1_model, stage1_scaler, stage2_model, metadata = load_artifacts()
    failure_types = metadata["failure_type_names"]
    feature_columns = metadata["feature_columns"]

    print(f"Loading raw readings from {args.input} ...")
    df_raw = pd.read_csv(args.input)

    # Machine ID is for human readability only — never fed to the models.
    # Pulled out before encoding, re-attached to the output afterward.
    has_machine_id = MACHINE_ID_COLUMN in df_raw.columns
    if has_machine_id:
        machine_ids = df_raw[MACHINE_ID_COLUMN].reset_index(drop=True)
        df_raw = df_raw.drop(columns=[MACHINE_ID_COLUMN])

    X_raw = encode_raw(df_raw, feature_columns)
    print(f"Scoring {len(X_raw)} machines through the cascade...")

    ranked = score_batch(X_raw, stage1_model, stage1_scaler, stage2_model, failure_types)

    if has_machine_id:
        # ranked keeps its ORIGINAL row index (score_batch no longer resets
        # it), so a positional .map() against that index lines IDs up
        # correctly even after sorting by priority.
        ranked[MACHINE_ID_COLUMN] = ranked.index.map(machine_ids.to_dict())
        ranked = ranked.reset_index(drop=True)
        ranked = ranked[[MACHINE_ID_COLUMN] + [c for c in ranked.columns if c != MACHINE_ID_COLUMN]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ranked.to_csv(args.output, index=False)

    n_flagged = int((ranked["stage1_flagged"] == 1).sum())
    print(f"\n{n_flagged} of {len(ranked)} machines flagged for maintenance.")
    print(f"Saved ranked schedule -> {args.output}")
    print("\nTop of the schedule:")
    print(ranked.head(5).to_string(index=False))


if __name__ == "__main__":
    main()