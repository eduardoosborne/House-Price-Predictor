# House Price Predictor

Modelo de Machine Learning que estima o preço de venda de casas a partir de 79 características do imóvel, treinado no dataset **Ames Housing** da competição do Kaggle [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**▶ Demo online:** [Hugging Face Space](https://huggingface.co/spaces/eduardoosborne/house-price-predictor) · **Modelo:** [Hugging Face Hub](https://huggingface.co/eduardoosborne/house-price-predictor) · **Kaggle:** score público `PREENCHER` (posição `PREENCHER`)

![demo](docs/demo.png)

## Resultados

Validação cruzada com 5 folds, usando a métrica oficial da competição (RMSE sobre o logaritmo do preço; quanto menor, melhor):

| Modelo | RMSE (log) |
|---|---|
| Ridge (regressão linear regularizada) | **0.1145** ± 0.0078 |
| Gradient Boosting | 0.1160 ± 0.0079 |
| Random Forest | 0.1390 ± 0.0092 |
| Rede Neural (MLP) | 0.1850 ± 0.0426 |

O preço final é um **ensemble** (média em escala log) dos dois melhores modelos. Um RMSE de ~0.115 em log corresponde, grosso modo, a um erro típico em torno de 11–12% do preço.

## Como funciona

1. **Limpeza:** remoção de 2 outliers documentados pelo autor do dataset (casas muito grandes vendidas por valores muito baixos).
2. **Pré-processamento** (`ColumnTransformer`): mediana para números faltantes, moda + One-Hot Encoding para categorias e normalização (`StandardScaler`) para os modelos sensíveis à escala.
3. **Alvo em log:** os preços são assimétricos (poucas casas muito caras); treinar em `log(1 + preço)` deixa a distribuição mais simétrica e alinha o treino com a métrica do Kaggle.
4. **Avaliação honesta:** todas as métricas vêm de validação cruzada, nunca de dados que o modelo já viu.
5. **Deploy:** demo em Gradio no Hugging Face Spaces e versão web em Flask + HTML/CSS.

## Como rodar

```bash
make setup     # cria o venv e instala as dependências
make train     # treina, gera results/metrics.csv e as submissões do Kaggle
make demo      # demo Gradio em http://127.0.0.1:7860
make web       # versão Flask em http://127.0.0.1:5000
```

## Estrutura

```
├── train.py          # treino, validação cruzada e submissões do Kaggle
├── app.py            # demo Gradio (Hugging Face Spaces)
├── flask_app.py      # versão web Flask
├── publish_hf.py     # publica modelo e demo no Hugging Face
├── models/           # modelos treinados + metadata.json
├── results/          # métricas e arquivos de submissão
├── templates/ static/
└── data/             # dataset Ames Housing (Kaggle)
```

**Tecnologias:** Python, pandas, NumPy, scikit-learn, Gradio, Flask, Hugging Face Hub.

---
Projeto desenvolvido na disciplina de Inteligência Artificial da FEUP (Faculdade de Engenharia da Universidade do Porto) e expandido depois. Autor: **Eduardo Oliveira Osborne** · [LinkedIn](https://www.linkedin.com/in/eduardo-osborne-35a22728b/)