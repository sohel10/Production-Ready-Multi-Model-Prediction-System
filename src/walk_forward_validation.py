import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from .config import PROCESSED_DATA_PATH


def load_daily():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["posting_date"] = pd.to_datetime(df["posting_date"], utc=True)
    df["posting_date"] = df["posting_date"].dt.tz_localize(None)

    daily = (
        df.groupby(pd.Grouper(key="posting_date", freq="D"))["price"]
        .mean()
        .reset_index()
    )

    daily = daily.dropna()
    daily.columns = ["date", "avg_price"]
    return daily


def create_lags(df, lags=[1, 7]):
    for lag in lags:
        df[f"lag_{lag}"] = df["avg_price"].shift(lag)
    return df.dropna()


def walk_forward_xgb(daily):
    df = create_lags(daily.copy())
    errors = []

    for i in range(20, len(df)-1):
        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        X_train = train.drop(["date", "avg_price"], axis=1)
        y_train = train["avg_price"]
        X_test = test.drop(["date", "avg_price"], axis=1)
        y_test = test["avg_price"]

        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        errors.append(mean_squared_error(y_test, pred))

    return np.sqrt(np.mean(errors))


def walk_forward_arima(daily):
    y = daily["avg_price"]
    errors = []

    for i in range(20, len(y)-1):
        train = y[:i]
        test = y[i:i+1]

        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=1)

        errors.append(mean_squared_error(test, pred))

    return np.sqrt(np.mean(errors))



if __name__ == "__main__":
    daily = load_daily()
    xgb_rmse = walk_forward_xgb(daily)
    arima_rmse = walk_forward_arima(daily)

    print("Walk-Forward XGBoost RMSE:", xgb_rmse)
    print("Walk-Forward ARIMA RMSE:", arima_rmse)

import pandas as pd

comparison = pd.DataFrame({
    "Model": ["ARIMA (1,1,1)", "XGBoost (Lag Features)"],
    "Walk-Forward RMSE": [arima_rmse, xgb_rmse]
})

print("\nModel Comparison:")
print(comparison)
