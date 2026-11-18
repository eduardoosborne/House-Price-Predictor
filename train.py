import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- dados
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

# Remove 2 outliers famosos do dataset (casas enormes vendidas barato),
# recomendados pelo próprio autor do dataset (De Cock, 2011).
train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300_000))]

y = train["SalePrice"]
X = train.drop(columns=["Id", "SalePrice"])
X_test = test.drop(columns=["Id"])

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols), ("cat", cat, cat_cols)])


def make_model(regressor, scale: bool) -> TransformedTargetRegressor:
    # Treina em log(1 + preço) e devolve a previsão já em dólares.
    pipe = Pipeline([("prep", make_preprocessor(scale)), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


CANDIDATES = {
    "ridge": make_model(Ridge(alpha=10.0), scale=True),
    "random_forest": make_model(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED), scale=False),
    "gradient_boosting": make_model(
        GradientBoostingRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                                  subsample=0.8, random_state=SEED), scale=False),
    "neural_network": make_model(
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, max_iter=2000,
                     early_stopping=True, random_state=SEED), scale=True),
}

# ---------------------------------------------------------------- avaliação
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
log_y = np.log1p(y)
rows = []

for name, model in CANDIDATES.items():
    # Avaliamos no espaço log para reproduzir a métrica do Kaggle.
    inner = model.regressor
    scores = -cross_val_score(inner, X, log_y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    rows.append({"modelo": name, "rmse_log_cv": scores.mean(), "desvio": scores.std()})
    print(f"{name:<18} RMSE(log) = {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X, y)
    joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)
    pd.DataFrame({"Id": test["Id"], "SalePrice": model.predict(X_test)}).to_csv(
        RESULTS_DIR / f"submission_{name}.csv", index=False)

# Ensemble: média das previsões em log dos dois melhores modelos
metrics = pd.DataFrame(rows).sort_values("rmse_log_cv")
top2 = metrics["modelo"].head(2).tolist()
blend = np.expm1(np.mean([np.log1p(joblib.load(MODELS_DIR / f"{m}.joblib").predict(X_test)) for m in top2], axis=0))
pd.DataFrame({"Id": test["Id"], "SalePrice": blend}).to_csv(RESULTS_DIR / "submission_ensemble.csv", index=False)

metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
(MODELS_DIR / "metadata.json").write_text(json.dumps({
    "best_model": metrics.iloc[0]["modelo"],
    "ensemble": top2,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    # Valores "típicos" usados pela demo para preencher campos que o usuário não informa
    "defaults": {**X[num_cols].median().to_dict(), **X[cat_cols].mode().iloc[0].to_dict()},
}, indent=2, default=float))

print("\nRanking:\n", metrics.to_string(index=False))
print(f"Ensemble salvo com: {top2}")
