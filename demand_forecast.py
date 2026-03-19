# demand_forecast.py

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "demand_forecast_data.csv"
MODEL_FILE = "demand_forecast_xgb_model.pkl"


# =========================================================
# LOAD DATA
# =========================================================
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "date",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "season",
        "temperature",
        "rain",
        "local_event",
        "past_demand_1day",
        "past_demand_7day",
        "target_demand",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def prepare_features(df: pd.DataFrame):
    feature_columns = [
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "season",
        "temperature",
        "rain",
        "local_event",
        "past_demand_1day",
        "past_demand_7day",
    ]

    X = df[feature_columns].copy()
    y = df["target_demand"].copy()

    categorical_features = ["day_of_week", "season"]
    numeric_features = [
        "is_weekend",
        "is_holiday",
        "temperature",
        "rain",
        "local_event",
        "past_demand_1day",
        "past_demand_7day",
    ]

    return X, y, categorical_features, numeric_features


# =========================================================
# TRAIN / TEST SPLIT (TIME-BASED)
# =========================================================
def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


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

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42
    )

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
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n===== MODEL EVALUATION =====")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": np.round(preds, 2)
    })

    print("\nSample predictions:")
    print(comparison.head(10).to_string(index=False))


# =========================================================
# TRAIN
# =========================================================
def train():
    print("Loading data...")
    df = load_data(DATA_FILE)

    train_df, test_df = time_based_split(df, test_size=0.2)

    X_train, y_train, categorical_features, numeric_features = prepare_features(train_df)
    X_test, y_test, _, _ = prepare_features(test_df)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    pipeline = build_pipeline(categorical_features, numeric_features)

    print("\nTraining XGBoost model...")
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
# PREDICT NEXT DAY DEMAND
# =========================================================
def predict_next_day(model, input_data: dict):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Train model
    model = train()

    # Example prediction
    # Example from your prompt:
    # Saturday, weekend=1, holiday=0, Spring, 20°C, no rain, event=1, yesterday=95, last_week=100
    example_input = {
        "day_of_week": "Saturday",
        "is_weekend": 1,
        "is_holiday": 0,
        "season": "Spring",
        "temperature": 20,
        "rain": 0,
        "local_event": 1,
        "past_demand_1day": 95,
        "past_demand_7day": 100,
    }

    pred = predict_next_day(model, example_input)

    print("\n===== EXAMPLE PREDICTION =====")
    print("Input:", example_input)
    print(f"Predicted target_demand = {pred}")