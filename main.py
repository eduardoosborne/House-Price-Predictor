import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================
# FILE PATHS
# =====================================================

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
SAMPLE_FILE = "data/sample_submission.csv"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATASETS
# =====================================================

print("Loading datasets...")

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

# =====================================================
# PREPARE TRAINING DATA
# =====================================================

y = train_data["SalePrice"]

X = train_data.drop(
    ["SalePrice", "Id"],
    axis=1
)

X_test = test_data.drop(
    ["Id"],
    axis=1
)

test_ids = test_data["Id"]

# =====================================================
# DETECT COLUMN TYPES
# =====================================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns

# =====================================================
# PREPROCESSING
# =====================================================

numerical_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_cols
        ),
        (
            "cat",
            categorical_transformer,
            categorical_cols
        )
    ]
)

# =====================================================
# TRAIN + PREDICT FUNCTION
# =====================================================

def run_model(model_name, regressor):

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    # -------------------------------------------------
    # Create pipeline
    # -------------------------------------------------

    model = Pipeline(steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            regressor
        )
    ])

    # -------------------------------------------------
    # Train model
    # -------------------------------------------------

    print("Training model...")

    model.fit(X, y)

    # -------------------------------------------------
    # Training evaluation
    # -------------------------------------------------

    train_predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        train_predictions
    )

    r2 = r2_score(
        y,
        train_predictions
    )

    print(f"MAE: {mae:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------

    model_file = os.path.join(
        RESULTS_DIR,
        f"{model_name}.pkl"
    )

    joblib.dump(model, model_file)

    print(f"Saved model: {model_file}")

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------

    print("Generating predictions...")

    predictions = model.predict(X_test)

    predictions = predictions.round().astype(int)

    # -------------------------------------------------
    # Save submission.csv
    # -------------------------------------------------

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions
    })

    submission_file = os.path.join(
        RESULTS_DIR,
        f"{model_name}_submission.csv"
    )

    submission.to_csv(
        submission_file,
        index=False
    )

    print(f"Saved predictions: {submission_file}")

    # -------------------------------------------------
    # Compare with sample submission
    # -------------------------------------------------

    sample_submission = pd.read_csv(
        SAMPLE_FILE
    )

    comparison_lines = []

    comparison_lines.append(
        f"MODEL: {model_name}"
    )

    comparison_lines.append("=" * 60)
    comparison_lines.append("")

    accuracies = []

    for i in range(
        min(20, len(submission))
    ):

        model_price = submission.loc[
            i,
            "SalePrice"
        ]

        sample_price = sample_submission.loc[
            i,
            "SalePrice"
        ]

        difference = (
            model_price - sample_price
        )

        if sample_price != 0:

            accuracy = 100 - (
                abs(
                    model_price - sample_price
                )
                / sample_price
                * 100
            )

        else:

            accuracy = 0

        accuracy = max(0, accuracy)

        accuracies.append(accuracy)

        comparison_lines.append(
            f"House ID {submission.loc[i, 'Id']}"
        )

        comparison_lines.append(
            f"  Model Prediction : ${model_price:,}"
        )

        comparison_lines.append(
            f"  Sample Prediction: ${sample_price:,}"
        )

        comparison_lines.append(
            f"  Difference       : ${difference:,}"
        )

        comparison_lines.append(
            f"  Similarity Score : {accuracy:.2f}%"
        )

        comparison_lines.append("")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    avg_model = submission[
        "SalePrice"
    ].mean()

    avg_sample = sample_submission[
        "SalePrice"
    ].mean()

    overall_accuracy = (
        sum(accuracies)
        / len(accuracies)
    )

    comparison_lines.append("=" * 60)
    comparison_lines.append("SUMMARY")
    comparison_lines.append("=" * 60)

    comparison_lines.append(
        f"Average Model Price : ${avg_model:,.2f}"
    )

    comparison_lines.append(
        f"Average Sample Price: ${avg_sample:,.2f}"
    )

    comparison_lines.append(
        f"Average Difference  : ${avg_model - avg_sample:,.2f}"
    )

    comparison_lines.append(
        f"Average Similarity  : {overall_accuracy:.2f}%"
    )

    comparison_lines.append("")
    comparison_lines.append(
        f"Training MAE        : {mae:,.2f}"
    )

    comparison_lines.append(
        f"Training R2 Score   : {r2:.4f}"
    )

    # -------------------------------------------------
    # Save comparison report
    # -------------------------------------------------

    comparison_file = os.path.join(
        RESULTS_DIR,
        f"{model_name}_report.txt"
    )

    with open(comparison_file, "w") as f:

        f.write(
            "\n".join(comparison_lines)
        )

    print(f"Saved report: {comparison_file}")

# =====================================================
# RANDOM FOREST
# =====================================================

run_model(
    "random_forest",
    RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
)

# =====================================================
# LINEAR REGRESSION
# =====================================================

run_model(
    "linear_regression",
    LinearRegression()
)

# =====================================================
# NEURAL NETWORK
# =====================================================

run_model(
    "neural_network",
    MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    )
)

# =====================================================
# FINISHED
# =====================================================

print("\nAll models completed successfully!")