"""
Lógica de previsão usada pela demo (app.py).
O usuário informa as 8 características que mais pesam no preço;
as outras 71 são preenchidas com valores típicos do dataset.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
META = json.loads((MODELS_DIR / "metadata.json").read_text())
NAMES = {"ridge": "Ridge", "gradient_boosting": "Gradient Boosting", "random_forest": "Random Forest"}

NEIGHBORHOODS = sorted([
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert",
    "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
])
QUALITY = {"Excelente": "Ex", "Boa": "Gd", "Média": "TA", "Razoável": "Fa"}
SQFT = 10.7639  # 1 m² = 10,76 pés²


def load_models() -> dict:
    return {key: joblib.load(MODELS_DIR / f"{key}.joblib") for key in NAMES}


def build_row(overall_qual, area_m2, basement_m2, garage_cars, year_built, full_bath,
              neighborhood, kitchen_qual) -> pd.DataFrame:
    """Monta a linha de 79 colunas que o modelo espera."""
    row = dict(META["defaults"])
    row.update({
        "OverallQual": overall_qual,
        "GrLivArea": area_m2 * SQFT,
        "TotalBsmtSF": basement_m2 * SQFT,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "YearRemodAdd": max(year_built, row["YearRemodAdd"]),
        "FullBath": full_bath,
        "Neighborhood": neighborhood,
        "KitchenQual": kitchen_qual,
        # Campos que "andam junto" com os informados, para a casa ficar coerente
        "ExterQual": kitchen_qual,
        "GarageArea": garage_cars * 270,
        "TotRmsAbvGrd": max(3, round(area_m2 / 22)),
    })
    first = min(row["GrLivArea"], max(row["TotalBsmtSF"], row["GrLivArea"] * 0.6))
    row["1stFlrSF"], row["2ndFlrSF"] = first, row["GrLivArea"] - first
    return pd.DataFrame([row])[META["num_cols"] + META["cat_cols"]]


def predict(models: dict, **features) -> tuple[float, dict]:
    """Devolve (preço do ensemble, previsões por modelo) em dólares."""
    df = build_row(**features)
    preds = {NAMES[key]: float(model.predict(df)[0]) for key, model in models.items()}
    final = float(np.expm1(np.mean([np.log1p(preds[NAMES[k]]) for k in META["ensemble"]])))
    return final, preds
