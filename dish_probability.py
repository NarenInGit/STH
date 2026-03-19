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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "dish_probability_data.csv"
MODEL_FILE = "dish_probability_xgb_model.pkl"
DEFAULT_JSON_FILE = "restaurant_forecast.json"


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

    numeric_columns = [
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

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


# =========================================================
# FEATURE CONFIG
# =========================================================
def get_feature_config():
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

    return feature_columns, target_columns, categorical_features, numeric_features


def prepare_features(df: pd.DataFrame):
    feature_columns, target_columns, categorical_features, numeric_features = get_feature_config()
    X = df[feature_columns].copy()
    y = df[target_columns].copy()
    return X, y, categorical_features, numeric_features


# =========================================================
# SPLITS
# =========================================================
def split_data(df: pd.DataFrame, test_size: float = 0.2):
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
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


# =========================================================
# NORMALIZE PROBABILITIES
# =========================================================
def normalize_probabilities(predictions: np.ndarray) -> np.ndarray:
    predictions = np.clip(predictions, 0, None)
    row_sums = predictions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return predictions / row_sums


# =========================================================
# MODEL
# =========================================================
def build_base_model():
    return XGBRegressor(
        n_estimators=1000,
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
    )


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(model, X_test_transformed, y_test):
    raw_preds = model.predict(X_test_transformed)
    preds = normalize_probabilities(raw_preds)
    y_true = y_test.values

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)

    print("\n===== DISH MODEL EVALUATION =====")
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
def train(data_file: str = DATA_FILE, model_file: str = MODEL_FILE):
    print("Loading dish probability data...")
    df = load_data(data_file)

    if len(df) < 30:
        raise ValueError("Dish dataset is too small. You should have at least around 30 rows.")

    train_df, test_df = split_data(df, test_size=0.2)
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

    # Train 3 separate XGBoost models so each can use early stopping
    models = {}
    target_names = ["dish1_probability", "dish2_probability", "dish3_probability"]

    print("\nTraining dish probability models...")
    for i, target_name in enumerate(target_names):
        model = build_base_model()
        model.set_params(early_stopping_rounds=50)
        model.fit(
            X_train_transformed,
            y_train.iloc[:, i],
            eval_set=[(X_val_transformed, y_val.iloc[:, i])],
            verbose=False,
        )
        models[target_name] = model

    evaluate_model(
        model=DishProbabilityEnsemble(models),
        X_test_transformed=X_test_transformed,
        y_test=y_test
    )

    artifact = {
        "preprocessor": preprocessor,
        "models": models,
        "feature_columns": get_feature_config()[0],
    }

    joblib.dump(artifact, model_file)
    print(f"\nDish model saved to: {model_file}")
    return artifact


# =========================================================
# WRAPPER FOR EVALUATION/PREDICTION
# =========================================================
class DishProbabilityEnsemble:
    def __init__(self, models: dict):
        self.models = models

    def predict(self, X):
        preds = []
        for key in ["dish1_probability", "dish2_probability", "dish3_probability"]:
            preds.append(self.models[key].predict(X))
        return np.column_stack(preds)


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
def predict_dish_probabilities(model_artifact, input_data: dict) -> dict:
    feature_columns = model_artifact["feature_columns"]
    preprocessor = model_artifact["preprocessor"]
    models = model_artifact["models"]

    input_df = pd.DataFrame([input_data])

    missing = [col for col in feature_columns if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing input fields for dish prediction: {missing}")

    input_df = input_df[feature_columns]
    X_transformed = preprocessor.transform(input_df)

    raw_preds = np.column_stack([
        models["dish1_probability"].predict(X_transformed),
        models["dish2_probability"].predict(X_transformed),
        models["dish3_probability"].predict(X_transformed),
    ])

    pred = normalize_probabilities(raw_preds)[0]

    return {
        "dish1_probability": round(float(pred[0]), 4),
        "dish2_probability": round(float(pred[1]), 4),
        "dish3_probability": round(float(pred[2]), 4),
    }


# =========================================================
# OPTIONAL COUNTS
# =========================================================
def estimate_dish_counts(probabilities: dict, predicted_total_demand: int) -> dict:
    return {
        "dish1_estimated_orders": int(round(probabilities["dish1_probability"] * predicted_total_demand)),
        "dish2_estimated_orders": int(round(probabilities["dish2_probability"] * predicted_total_demand)),
        "dish3_estimated_orders": int(round(probabilities["dish3_probability"] * predicted_total_demand)),
    }


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


def predict_dish_result(model_artifact, input_data: dict) -> dict:
    probs = predict_dish_probabilities(model_artifact, input_data)
    now = datetime.now()

    result = _base_result_dict(now)
    result["dish1_probability"] = float(probs["dish1_probability"])
    result["dish2_probability"] = float(probs["dish2_probability"])
    result["dish3_probability"] = float(probs["dish3_probability"])
    return result


def save_dish_result_to_json(
    model_artifact,
    input_data: dict,
    output_file: str = DEFAULT_JSON_FILE
) -> dict:
    result = read_or_create_result_json(output_file)
    probs = predict_dish_probabilities(model_artifact, input_data)
    now = datetime.now()

    result["date"] = now.strftime("%Y-%m-%d")
    result["time"] = now.strftime("%H:%M:%S")
    result["dish1_probability"] = float(probs["dish1_probability"])
    result["dish2_probability"] = float(probs["dish2_probability"])
    result["dish3_probability"] = float(probs["dish3_probability"])

    write_result_json(result, output_file)
    print(f"Dish JSON saved to: {output_file}")
    return result


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    model_artifact = train()

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

    probs = predict_dish_probabilities(model_artifact, example_input)
    counts = estimate_dish_counts(probs, example_input["predicted_total_demand"])

    print("\n===== EXAMPLE DISH PREDICTION =====")
    print("Input:", example_input)
    print("Predicted probabilities:", probs)
    print("Estimated dish counts:", counts)

    save_dish_result_to_json(
        model_artifact=model_artifact,
        input_data=example_input,
        output_file=DEFAULT_JSON_FILE
    )
