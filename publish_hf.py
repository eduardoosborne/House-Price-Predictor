"""
Publica (ou atualiza) o modelo no Hugging Face Model Hub, com model card.
A demo em Gradio (app.py) roda localmente: desde 2026, hospedar Spaces com Gradio
no Hugging Face exige o plano PRO.

Antes, faça login uma vez no terminal:  hf auth login
(crie um token com permissão "write" em https://huggingface.co/settings/tokens)

Uso:  python publish_hf.py
"""
import pandas as pd
from huggingface_hub import HfApi

MODEL_NAME = "house-price-predictor"
GITHUB_URL = "https://github.com/eduardoosborne/House-Price-Predictor"

api = HfApi()
user = api.whoami()["name"]
model_id = f"{user}/{MODEL_NAME}"

metrics = pd.read_csv("results/metrics.csv")
table = "\n".join(f"| {r.modelo} | {r.rmse_log_cv:.4f} ± {r.desvio:.4f} |" for r in metrics.itertuples())

MODEL_CARD = f"""---
license: mit
library_name: sklearn
tags: [tabular-regression, sklearn, house-prices, kaggle]
datasets: [ames-housing]
metrics: [rmse]
---
# House Price Predictor (Ames Housing)

Modelos scikit-learn que estimam o preço de venda de casas em Ames (Iowa, EUA),
treinados no dataset da competição
[House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

- **Código:** {GITHUB_URL}

## Resultados (validação cruzada 5-fold, RMSE sobre log do preço; menor é melhor)

| Modelo | RMSE (log) |
|---|---|
{table}

O preço final da demo é a média (em escala log) dos dois melhores modelos.

## Como usar

```python
import joblib, pandas as pd
from huggingface_hub import hf_hub_download

model = joblib.load(hf_hub_download("{model_id}", "models/gradient_boosting.joblib"))
# df: DataFrame com as 79 colunas do dataset (veja models/metadata.json)
preco = model.predict(df)
```

## Limitações
- Dados de uma única cidade americana vendidos entre 2006 e 2010: não generaliza para outros mercados.
- A demo (`app.py`) pede só 8 características e preenche as demais com valores típicos, então é uma estimativa aproximada.

Projeto da disciplina de Inteligência Artificial na FEUP (Universidade do Porto), por Eduardo Oliveira Osborne.
"""

api.create_repo(model_id, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=model_id, folder_path=".", allow_patterns=["models/*", "results/metrics.csv"],
                  commit_message="Upload models")
api.upload_file(repo_id=model_id, path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md")
print(f"Modelo publicado: https://huggingface.co/{model_id}")
