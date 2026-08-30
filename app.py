from flask import Flask, request, jsonify, render_template
import pandas as pd

from src.predict_batch import load_artifacts, encode_raw, score_batch, MACHINE_ID_COLUMN

app = Flask(__name__)

# Load everything ONCE when the server starts
stage1_model, stage1_scaler, stage2_model, metadata = load_artifacts()
failure_types = metadata["failure_type_names"]
feature_columns = metadata["feature_columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded"}), 400

    df_raw = pd.read_csv(file)

    # Same Machine ID handling as predict_batch.py's main()
    has_machine_id = MACHINE_ID_COLUMN in df_raw.columns
    if has_machine_id:
        machine_ids = df_raw[MACHINE_ID_COLUMN].reset_index(drop=True)
        df_raw = df_raw.drop(columns=[MACHINE_ID_COLUMN])

    try:
        X_raw = encode_raw(df_raw, feature_columns)
        ranked = score_batch(X_raw, stage1_model, stage1_scaler, stage2_model, failure_types)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if has_machine_id:
        ranked[MACHINE_ID_COLUMN] = ranked.index.map(machine_ids.to_dict())
        ranked = ranked.reset_index(drop=True)

    # Convert the table into a list of dictionaries — JSON-friendly
    ranked = ranked.astype(object).where(ranked.notnull(), None)
    return jsonify(ranked.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)