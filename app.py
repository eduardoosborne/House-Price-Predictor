"""
Demo local em Gradio.
Rode com: make demo   (abre em http://127.0.0.1:7860)
"""
import gradio as gr
import pandas as pd

from predictor import NEIGHBORHOODS, QUALITY, load_models, predict

MODELS = load_models()


def run(overall_qual, area_m2, basement_m2, garage_cars, year_built, full_bath, neighborhood, kitchen_qual):
    final, preds = predict(
        MODELS, overall_qual=overall_qual, area_m2=area_m2, basement_m2=basement_m2,
        garage_cars=garage_cars, year_built=year_built, full_bath=full_bath,
        neighborhood=neighborhood, kitchen_qual=kitchen_qual,
    )
    table = pd.DataFrame({"Modelo": list(preds), "Previsão (US$)": [f"{v:,.0f}" for v in preds.values()]})
    return f"US$ {final:,.0f}", table


with gr.Blocks(title="House Price Predictor") as demo:
    gr.Markdown(
        "# House Price Predictor\n"
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
            kitchen = gr.Dropdown(list(QUALITY.items()), value="TA", label="Qualidade da cozinha")
            button = gr.Button("Estimar preço", variant="primary")
        with gr.Column():
            price = gr.Textbox(label="Preço estimado (ensemble)")
            detail = gr.Dataframe(label="Previsão por modelo")
    button.click(run, [overall_qual, area, basement, garage, year, bath, hood, kitchen], [price, detail])

if __name__ == "__main__":
    demo.launch()
