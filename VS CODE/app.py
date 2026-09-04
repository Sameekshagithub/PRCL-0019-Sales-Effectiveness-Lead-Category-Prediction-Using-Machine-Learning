"""
PRCL-0019 : Sales Effectiveness — Flask Web Application
---------------------------------------------------------
Serves a web form where a user enters a new lead's details and gets back
the predicted Lead Category (High Potential / Low Potential).

Run train_model.py first to generate model_artifacts/. Then run:

    python app.py

and open http://127.0.0.1:5000 in your browser.
"""

import os

import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

ARTIFACT_DIR = "model_artifacts"
required_files = [
    "lead_model.pkl", "scaler.pkl", "label_encoder.pkl",
    "feature_columns.pkl", "needs_scaling.pkl", "form_options.pkl",
]
missing = [f for f in required_files if not os.path.exists(os.path.join(ARTIFACT_DIR, f))]
if missing:
    raise FileNotFoundError(
        f"Missing model artifacts: {missing}. Run `python train_model.py` first."
    )

lead_model = joblib.load(os.path.join(ARTIFACT_DIR, "lead_model.pkl"))
scaler_obj = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
label_enc = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))
needs_scaling = joblib.load(os.path.join(ARTIFACT_DIR, "needs_scaling.pkl"))
options = joblib.load(os.path.join(ARTIFACT_DIR, "form_options.pkl"))


def build_feature_row(form):
    """Turn the submitted form fields into a single row matching the exact
    feature layout the model was trained on."""
    created = pd.to_datetime(form["Created"])

    raw = {
        "Product_ID": float(form["Product_ID"]),
        "Created_Year": created.year,
        "Created_Month": created.month,
        "Created_Day": created.day,
        "Has_Mobile": int(form["Has_Mobile"]),
        "Has_Email": int(form["Has_Email"]),
        "Source": form["Source"],
        "Sales_Agent": form["Sales_Agent"],
        "Location": form["Location"],
        "Delivery_Mode": form["Delivery_Mode"],
    }

    row_df = pd.DataFrame([raw])
    row_encoded = pd.get_dummies(
        row_df, columns=["Source", "Sales_Agent", "Location", "Delivery_Mode"]
    )

    # Align to training-time columns; anything not present for this single
    # row (a category that wasn't selected) is filled with 0
    row_encoded = row_encoded.reindex(columns=feature_cols, fill_value=0)

    if needs_scaling:
        row_encoded = scaler_obj.transform(row_encoded)

    return row_encoded


@app.route("/")
def home():
    return render_template("index.html", options=options, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    features = build_feature_row(request.form)
    pred_encoded = lead_model.predict(features)[0]
    pred_label = label_enc.inverse_transform([pred_encoded])[0]
    return render_template("index.html", options=options, prediction=pred_label)


if __name__ == "__main__":
    app.run(debug=True)
