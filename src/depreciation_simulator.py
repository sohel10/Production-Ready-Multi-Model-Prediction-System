import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from .preprocessing import preprocess_pipeline


# --------------------------------------------------
# 1️⃣ Validation Utilities
# --------------------------------------------------

def validate_dataset(df):

    print("\n===== DATA VALIDATION =====")
    print("Total rows:", len(df))

    if "residual_value_pct" not in df.columns:
        raise ValueError("❌ residual_value_pct column missing.")

    print("Missing RV%:", df["residual_value_pct"].isna().sum())

    print("RV% min:", df["residual_value_pct"].min())
    print("RV% max:", df["residual_value_pct"].max())

    # Sanity bounds
    if df["residual_value_pct"].max() > 1.2:
        print("⚠ Warning: Some RV% > 120%")

    if df["residual_value_pct"].min() < 0:
        print("⚠ Warning: Negative RV% detected")


# --------------------------------------------------
# 2️⃣ Segment Builder
# --------------------------------------------------

def get_segments(df):

    return {
        "truck": df[df["type"] == "truck"],
        "SUV": df[df["type"].str.lower() == "suv"],
        "sedan": df[df["type"] == "sedan"],
        "luxury": df[df["manufacturer"].isin(["bmw", "lexus", "audi"])],
        "ev": df[df["fuel"] == "electric"]
    }


# --------------------------------------------------
# 3️⃣ Core Training Function
# --------------------------------------------------

def train_segment_model(segment_df, segment_name):

    if len(segment_df) < 500:
        print(f"⚠ Skipping {segment_name} (too small)")
        return None

    segment_df = segment_df.sort_values("posting_date")

    train_df = segment_df[segment_df["posting_date"] < "2024-01-01"]
    test_df  = segment_df[segment_df["posting_date"] >= "2024-01-01"]

    if len(train_df) == 0 or len(test_df) == 0:
        print(f"⚠ Skipping {segment_name} (empty split)")
        return None

    feature_cols = [
        "vehicle_age",
        "log_odometer",
        "age_x_log_odometer",
        "price_per_mile"
    ]

    categorical_cols = [
        "manufacturer",
        "state",
        "transmission",
        "drive",
        "fuel"
    ]

    X_train = pd.get_dummies(
        train_df[feature_cols + categorical_cols],
        drop_first=True
    )

    X_test = pd.get_dummies(
        test_df[feature_cols + categorical_cols],
        drop_first=True
    )

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df["residual_value_pct"]
    y_test  = test_df["residual_value_pct"]

    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n🚗 {segment_name.upper()} RMSE (RV%): {rmse:.4f}")

    return model


# --------------------------------------------------
# 4️⃣ Main Runner
# --------------------------------------------------

def run_residual_system():

    df = preprocess_pipeline()

    validate_dataset(df)

    segments = get_segments(df)

    trained_models = {}

    for name, segment_df in segments.items():
        model = train_segment_model(segment_df, name)
        if model:
            trained_models[name] = model

    print("\n✅ Residual System Training Completed")

    return trained_models


if __name__ == "__main__":
    run_residual_system()
