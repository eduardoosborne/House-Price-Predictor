"""Versão web original (Flask + HTML). Rode com: python flask_app.py"""
import json
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

META = json.loads(open("models/metadata.json").read())
MODELS = {
    "Ridge": joblib.load("models/ridge.joblib"),
    "Gradient Boosting": joblib.load("models/gradient_boosting.joblib"),
    "Random Forest": joblib.load("models/random_forest.joblib"),
}
ENSEMBLE = {"ridge": "Ridge", "gradient_boosting": "Gradient Boosting", "random_forest": "Random Forest"}


def to_frame(data: dict) -> pd.DataFrame:
    """Completa campos ausentes com valores típicos e converte números."""
    row = dict(META["defaults"])
    for key, value in data.items():
        if value in ("", None):
            continue
        row[key] = float(value) if key in META["num_cols"] else value
    return pd.DataFrame([row])[META["num_cols"] + META["cat_cols"]]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid or empty JSON body"}), 400
        df = to_frame(data)
        preds = {name: round(float(m.predict(df)[0]), 2) for name, m in MODELS.items()}
        best = [ENSEMBLE[m] for m in META["ensemble"]]
        final = float(np.expm1(np.mean([np.log1p(preds[n]) for n in best])))
        return jsonify({"predictions": preds, "average": round(final, 2)})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
