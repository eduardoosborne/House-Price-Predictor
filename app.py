"""
Demo para o Hugging Face Spaces
O usuário informa as 8 características que mais pesam no preço;
as outras 71 são preenchidas com valores típicos do dataset.
Rode localmente com: python app.py
"""
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd

META = json.loads(open("models/metadata.json").read())
NAMES = {"ridge": "Ridge", "gradient_boosting": "Gradient Boosting", "random_forest": "Random Forest"}
MODELS = {key: joblib.load(f"models/{key}.joblib") for key in NAMES}

NEIGHBORHOODS = sorted([
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert",
    "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
])
QUALITY = [("Excelente", "Ex"), ("Boa", "Gd"), ("Média", "TA"), ("Razoável", "Fa")]
SQFT = 10.7639  # 1 m² = 10,76 pés²


def predict(overall_qual, area_m2, basement_m2, garage_cars, year_built, full_bath, neighborhood, kitchen_qual):
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
    df = pd.DataFrame([row])[META["num_cols"] + META["cat_cols"]]
    preds = {key: float(model.predict(df)[0]) for key, model in MODELS.items()}
    final = float(np.expm1(np.mean([np.log1p(preds[k]) for k in META["ensemble"]])))

    table = pd.DataFrame({
        "Modelo": [NAMES[k] for k in preds],
        "Previsão (US$)": [f"{v:,.0f}" for v in preds.values()],
    })
    return f"US$ {final:,.0f}", table


with gr.Blocks(title="House Price Predictor") as demo:
    gr.Markdown(
        "# 🏠 House Price Predictor\n"
        "Estima o preço de uma casa em Ames (Iowa, EUA) com modelos treinados no dataset "
        "[Ames Housing do Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). "
        "O preço final é um *ensemble* dos dois melhores modelos na validação cruzada."
    )
    with gr.Row():
        with gr.Column():
            overall_qual = gr.Slider(1, 10, value=6, step=1, label="Qualidade geral da construção (1–10)")
            area = gr.Slider(30, 450, value=140, step=5, label="Área habitável acima do solo (m²)")
            basement = gr.Slider(0, 300, value=90, step=5, label="Área do porão (m²)")
            garage = gr.Slider(0, 4, value=2, step=1, label="Vagas na garagem")
            year = gr.Slider(1872, 2010, value=1975, step=1, label="Ano de construção")
            bath = gr.Slider(0, 4, value=2, step=1, label="Banheiros completos")
            hood = gr.Dropdown(NEIGHBORHOODS, value="NAmes", label="Bairro")
            kitchen = gr.Dropdown(QUALITY, value="TA", label="Qualidade da cozinha")
            button = gr.Button("Estimar preço", variant="primary")
        with gr.Column():
            price = gr.Textbox(label="Preço estimado (ensemble)")
            detail = gr.Dataframe(label="Previsão por modelo")
    button.click(predict, [overall_qual, area, basement, garage, year, bath, hood, kitchen], [price, detail])

if __name__ == "__main__":
    demo.launch()
