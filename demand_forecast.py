import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "demand_forecast_data.csv"
MODEL_FILE = "demand_forecast_xgb_model.pkl"
DEFAULT_JSON_FILE = "restaurant_forecast.json"


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

    numeric_columns = [
        "is_weekend",
        "is_holiday",
        "temperature",
        "rain",
        "local_event",
        "past_demand_1day",
        "past_demand_7day",
        "target_demand",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def get_feature_config():
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

    return feature_columns, categorical_features, numeric_features


def prepare_features(df: pd.DataFrame):
    feature_columns, categorical_features, numeric_features = get_feature_config()
    X = df[feature_columns].copy()
    y = df["target_demand"].copy()
    return X, y, categorical_features, numeric_features


# =========================================================
# SPLITS
# =========================================================
def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def split_train_validation(train_df: pd.DataFrame, val_size: float = 0.15):
    split_index = int(len(train_df) * (1 - val_size))
    train_part = train_df.iloc[:split_index].copy()
    val_part = train_df.iloc[split_index:].copy()
    return train_part, val_part


# =========================================================
# PREPROCESSOR
# =========================================================
def build_preprocessor(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    return preprocessor


# =========================================================
# MODEL
# =========================================================
def build_model():
    # More conservative settings to reduce overfitting
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.2,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    return model


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(model, X_test_transformed, y_test):
    preds = model.predict(X_test_transformed)
    preds = np.maximum(preds, 0)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n===== DEMAND MODEL EVALUATION =====")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    comparison = pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": np.round(preds[:10], 2)
    })

    print("\nSample predictions:")
    print(comparison.to_string(index=False))


# =========================================================
# TRAIN
# =========================================================
def train(data_file: str = DATA_FILE, model_file: str = MODEL_FILE):
    print("Loading demand data...")
    df = load_data(data_file)

    if len(df) < 30:
        raise ValueError("Demand dataset is too small. You should have at least around 30 rows.")

    train_df, test_df = time_based_split(df, test_size=0.2)
    train_part, val_part = split_train_validation(train_df, val_size=0.15)

    X_train, y_train, categorical_features, numeric_features = prepare_features(train_part)
    X_val, y_val, _, _ = prepare_features(val_part)
    X_test, y_test, _, _ = prepare_features(test_df)

    print(f"Train rows      : {len(train_part)}")
    print(f"Validation rows : {len(val_part)}")
    print(f"Test rows       : {len(test_df)}")

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    model = build_model()

    print("\nTraining demand model...")
    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_val_transformed, y_val)],
        verbose=False,
    )

    evaluate_model(model, X_test_transformed, y_test)

    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": get_feature_config()[0],
    }

    joblib.dump(artifact, model_file)
    print(f"\nDemand model saved to: {model_file}")
    return artifact


# =========================================================
# LOAD SAVED MODEL
# =========================================================
def load_model(model_path: str = MODEL_FILE):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


# =========================================================
# PREDICTION
# =========================================================
def predict_next_day(model_artifact, input_data: dict) -> float:
    feature_columns = model_artifact["feature_columns"]
    preprocessor = model_artifact["preprocessor"]
    model = model_artifact["model"]

    input_df = pd.DataFrame([input_data])

    missing = [col for col in feature_columns if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing input fields for demand prediction: {missing}")

    input_df = input_df[feature_columns]
    X_transformed = preprocessor.transform(input_df)
    prediction = model.predict(X_transformed)[0]
    prediction = max(float(prediction), 0.0)

    return round(prediction, 2)


# =========================================================
# JSON HELPERS
# =========================================================
def _base_result_dict(now: datetime | None = None) -> dict:
    now = now or datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "forecasted_demand": None,
        "dish1_probability": None,
        "dish2_probability": None,
        "dish3_probability": None,
    }


def read_or_create_result_json(output_file: str = DEFAULT_JSON_FILE) -> dict:
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = _base_result_dict()
            base.update(data if isinstance(data, dict) else {})
            return base
        except Exception:
            return _base_result_dict()
    return _base_result_dict()


def write_result_json(result: dict, output_file: str = DEFAULT_JSON_FILE) -> dict:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    return result


def predict_demand_result(model_artifact, input_data: dict) -> dict:
    forecasted_demand = int(round(predict_next_day(model_artifact, input_data)))
    now = datetime.now()

    result = _base_result_dict(now)
    result["forecasted_demand"] = forecasted_demand
    return result


def save_demand_result_to_json(
    model_artifact,
    input_data: dict,
    output_file: str = DEFAULT_JSON_FILE
) -> dict:
    result = read_or_create_result_json(output_file)
    now = datetime.now()

    result["date"] = now.strftime("%Y-%m-%d")
    result["time"] = now.strftime("%H:%M:%S")
    result["forecasted_demand"] = int(round(predict_next_day(model_artifact, input_data)))

    write_result_json(result, output_file)
    print(f"Demand JSON saved to: {output_file}")
    return result


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    model_artifact = train()

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

    pred = predict_next_day(model_artifact, example_input)

    print("\n===== EXAMPLE DEMAND PREDICTION =====")
    print("Input:", example_input)
    print(f"Predicted target_demand = {pred}")

    save_demand_result_to_json(
        model_artifact=model_artifact,
        input_data=example_input,
        output_file=DEFAULT_JSON_FILE
    )
