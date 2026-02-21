import pandas as pd
from .config import PROCESSED_DATA_PATH

def load_cleaned_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df

def aggregate_daily_price(df):
    df["posting_date"] = pd.to_datetime(df["posting_date"], utc=True)
    
    daily = (
        df.groupby(df["posting_date"].dt.date)["price"]
        .mean()
        .reset_index()
    )
    
    daily.columns = ["date", "avg_price"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    
    return daily
def create_lag_features(df, lags=[1, 2, 3, 7]):
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["avg_price"].shift(lag)

    df = df.dropna()
    return df


if __name__ == "__main__":
    df = load_cleaned_data()
    daily_original = aggregate_daily_price(df)
    
    print("Original daily count:", len(daily_original))
    
    daily = create_lag_features(daily_original)
    print("After lag drop:", len(daily))

