import traceback
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

models = {
    "Random Forest":     joblib.load("results/random_forest.pkl"),
    "Linear Regression": joblib.load("results/linear_regression.pkl"),
    "Neural Network":    joblib.load("results/neural_network.pkl"),
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)

        if data is None:
            return jsonify({"error": "Invalid or empty JSON body"}), 400

        numeric_fields = [
            "MSSubClass", "LotFrontage", "LotArea", "OverallQual", "OverallCond",
            "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
            "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
            "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
            "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
            "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
            "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
            "MoSold", "YrSold"
        ]

        for field in numeric_fields:
            if field in data and data[field] != "":
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError):
                    data[field] = None

        df = pd.DataFrame([data])

        predictions = {}
        for name, model in models.items():
            predictions[name] = round(float(model.predict(df)[0]), 2)

        average = round(sum(predictions.values()) / len(predictions), 2)

        return jsonify({"predictions": predictions, "average": average})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
