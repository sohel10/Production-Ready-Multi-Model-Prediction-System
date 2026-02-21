import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from .config import PROCESSED_DATA_PATH


def load_daily():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Force proper datetime conversion
    df["posting_date"] = pd.to_datetime(
        df["posting_date"],
        errors="coerce",
        utc=True
    )

    # Drop rows where date failed to parse
    df = df.dropna(subset=["posting_date"])

    # Remove timezone info (optional but cleaner)
    df["posting_date"] = df["posting_date"].dt.tz_localize(None)

    daily = (
        df.groupby(df["posting_date"].dt.date)["price"]
        .mean()
        .reset_index()
    )

    daily.columns = ["date", "avg_price"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    return daily



def run_arima():
    daily = load_daily()

    y = daily["avg_price"]

    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]

    # Simple ARIMA(1,1,1)
    model = ARIMA(y_train, order=(1,1,1))
    model_fit = model.fit()

    preds = model_fit.forecast(steps=len(y_test))

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    print("Train size:", len(y_train))
    print("Test size:", len(y_test))
    print("ARIMA RMSE:", rmse)


if __name__ == "__main__":
    run_arima()
