# dish_probability.py

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "dish_probability_data.csv"
MODEL_FILE = "dish_probability_xgb_model.pkl"


# =========================================================
# LOAD DATA
# =========================================================
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "day_of_week",
        "season",
        "temperature",
        "rain",
        "local_event",
        "predicted_total_demand",
        "dish1_sales_yesterday",
        "dish2_sales_yesterday",
        "dish3_sales_yesterday",
        "dish1_probability",
        "dish2_probability",
        "dish3_probability",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# =========================================================
# FEATURE / TARGET PREP
# =========================================================
def prepare_features(df: pd.DataFrame):
    feature_columns = [
        "day_of_week",
        "season",
        "temperature",
        "rain",
        "local_event",
        "predicted_total_demand",
        "dish1_sales_yesterday",
        "dish2_sales_yesterday",
        "dish3_sales_yesterday",
    ]

    target_columns = [
        "dish1_probability",
        "dish2_probability",
        "dish3_probability",
    ]

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    categorical_features = ["day_of_week", "season"]
    numeric_features = [
        "temperature",
        "rain",
        "local_event",
        "predicted_total_demand",
        "dish1_sales_yesterday",
        "dish2_sales_yesterday",
        "dish3_sales_yesterday",
    ]

    return X, y, categorical_features, numeric_features


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
def split_data(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


# =========================================================
# NORMALIZE PROBABILITIES
# Ensures outputs sum to 1.0
# =========================================================
def normalize_probabilities(predictions: np.ndarray) -> np.ndarray:
    predictions = np.clip(predictions, 0, None)
    row_sums = predictions.sum(axis=1, keepdims=True)

    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0

    normalized = predictions / row_sums
    return normalized


# =========================================================
# BUILD MODEL
# =========================================================
def build_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    base_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )

    model = MultiOutputRegressor(base_model)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(model, X_test, y_test):
    raw_preds = model.predict(X_test)
    preds = normalize_probabilities(raw_preds)

    y_true = y_test.values

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print("\n===== MODEL EVALUATION =====")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    comparison = pd.DataFrame({
        "Actual_Dish1": np.round(y_true[:10, 0], 4),
        "Pred_Dish1": np.round(preds[:10, 0], 4),
        "Actual_Dish2": np.round(y_true[:10, 1], 4),
        "Pred_Dish2": np.round(preds[:10, 1], 4),
        "Actual_Dish3": np.round(y_true[:10, 2], 4),
        "Pred_Dish3": np.round(preds[:10, 2], 4),
    })

    print("\nSample predictions:")
    print(comparison.to_string(index=False))


# =========================================================
# TRAIN
# =========================================================
def train():
    print("Loading data...")
    df = load_data(DATA_FILE)

    train_df, test_df = split_data(df, test_size=0.2)

    X_train, y_train, categorical_features, numeric_features = prepare_features(train_df)
    X_test, y_test, _, _ = prepare_features(test_df)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    pipeline = build_pipeline(categorical_features, numeric_features)

    print("\nTraining XGBoost multi-output model...")
    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_test, y_test)

    print(f"\nSaving model to: {MODEL_FILE}")
    joblib.dump(pipeline, MODEL_FILE)

    print("Training complete.")
    return pipeline


# =========================================================
# LOAD SAVED MODEL
# =========================================================
def load_model(model_path: str = MODEL_FILE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


# =========================================================
# PREDICT DISH PROBABILITIES
# =========================================================
def predict_dish_probabilities(model, input_data: dict):
    input_df = pd.DataFrame([input_data])
    raw_pred = model.predict(input_df)
    pred = normalize_probabilities(raw_pred)[0]

    return {
        "dish1_probability": round(float(pred[0]), 4),
        "dish2_probability": round(float(pred[1]), 4),
        "dish3_probability": round(float(pred[2]), 4),
    }


# =========================================================
# ESTIMATE DISH COUNTS
# Converts probabilities into expected dish quantities
# =========================================================
def estimate_dish_counts(probabilities: dict, predicted_total_demand: int):
    return {
        "dish1_estimated_orders": round(probabilities["dish1_probability"] * predicted_total_demand),
        "dish2_estimated_orders": round(probabilities["dish2_probability"] * predicted_total_demand),
        "dish3_estimated_orders": round(probabilities["dish3_probability"] * predicted_total_demand),
    }


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    model = train()

    # Example from your prompt:
    # Saturday, Spring, 20C, no rain, event=1, demand=120, dish1=40, dish2=25, dish3=15
    example_input = {
        "day_of_week": "Saturday",
        "season": "Spring",
        "temperature": 20,
        "rain": 0,
        "local_event": 1,
        "predicted_total_demand": 120,
        "dish1_sales_yesterday": 40,
        "dish2_sales_yesterday": 25,
        "dish3_sales_yesterday": 15,
    }

    probs = predict_dish_probabilities(model, example_input)
    counts = estimate_dish_counts(probs, example_input["predicted_total_demand"])

    print("\n===== EXAMPLE PREDICTION =====")
    print("Input:", example_input)
    print("Predicted probabilities:", probs)
    print("Estimated dish counts:", counts)