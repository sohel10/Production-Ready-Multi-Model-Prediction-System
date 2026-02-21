import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os




# --------------------------------------------------
# 1️⃣ Create Lag Features
# --------------------------------------------------

def create_lag_features(df):

    df = df.copy().sort_values("year_week")

    df["lag_1"] = df["residual_value_pct"].shift(1)
    df["lag_2"] = df["residual_value_pct"].shift(2)
    df["lag_3"] = df["residual_value_pct"].shift(3)
    df["rolling_3"] = df["residual_value_pct"].rolling(3).mean()

    return df.dropna()


# --------------------------------------------------
# 2️⃣ Walk-Forward Validation
# --------------------------------------------------

def walk_forward_validation(df, model):

    if len(df) < 8:
        print("⚠️ Not enough data for walk-forward validation.")
        return None

    errors = []
    history = df.iloc[:4].copy()

    for i in range(4, len(df)):

        train = history.copy()
        test = df.iloc[i:i+1]

        X_train = train[["lag_1", "lag_2", "lag_3", "rolling_3"]]
        y_train = train["residual_value_pct"]

        model.fit(X_train, y_train)

        X_test = test[["lag_1", "lag_2", "lag_3", "rolling_3"]]
        y_test = test["residual_value_pct"]

        pred = model.predict(X_test)[0]
        errors.append((y_test.values[0] - pred) ** 2)

        history = pd.concat([history, test])

    if len(errors) == 0:
        return None

    return np.sqrt(np.mean(errors))

# --------------------------------------------------
# 3️⃣ Train Time Model (XGBoost + MLflow)
# --------------------------------------------------
from .config import TIME_MODEL_CONFIG
def train_time_model(weekly_df):

    df = create_lag_features(weekly_df)

    if len(df) < 10:
        print("⚠️ Limited weekly history — training without validation.")
    else:
        rmse = walk_forward_validation(df, xgb.XGBRegressor())
        if rmse is not None:
            print(f"Walk-forward RMSE: {rmse:.4f}")

    model = xgb.XGBRegressor(
    **TIME_MODEL_CONFIG,
    random_state=42,
    n_jobs=-1
    )

    X = df[["lag_1","lag_2","lag_3","rolling_3"]]
    y = df["residual_value_pct"]

    model.fit(X, y)

    return model


# --------------------------------------------------
# 4️⃣ Forecast Future Steps
# --------------------------------------------------

def forecast_future(model, weekly_df, steps=2):

    df = create_lag_features(weekly_df)

    last_values = df.iloc[-1].copy()
    forecasts = []

    for _ in range(steps):

        X_pred = pd.DataFrame([{
            "lag_1": last_values["lag_1"],
            "lag_2": last_values["lag_2"],
            "lag_3": last_values["lag_3"],
            "rolling_3": last_values["rolling_3"]
        }])

        pred = model.predict(X_pred)[0]
        forecasts.append(pred)

        # Update lags dynamically
        last_values["lag_3"] = last_values["lag_2"]
        last_values["lag_2"] = last_values["lag_1"]
        last_values["lag_1"] = pred
        last_values["rolling_3"] = np.mean([
            last_values["lag_1"],
            last_values["lag_2"],
            last_values["lag_3"]
        ])

    return forecasts