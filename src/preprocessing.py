import pandas as pd
import numpy as np
from datetime import datetime
from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH


# --------------------------------------------------
# 1️⃣ Load Data
# --------------------------------------------------
def load_data():
    return pd.read_csv(RAW_DATA_PATH)


# --------------------------------------------------
# 2️⃣ Basic Cleaning
# --------------------------------------------------
def clean_data(df):

    df["posting_date"] = pd.to_datetime(
        df["posting_date"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=["posting_date", "price", "year"])
    df = df[df["price"] > 0]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

    df = df.dropna(subset=["year"])

    return df


# --------------------------------------------------
# 3️⃣ Outlier Filtering + Vehicle Age
# --------------------------------------------------
def filter_outliers(df):

    df = df[df["price"] <= 100000]
    df = df[df["odometer"] <= 500000]

    current_year = datetime.now().year
    df["vehicle_age"] = current_year - df["year"]

    df = df[(df["vehicle_age"] >= 0) & (df["vehicle_age"] <= 50)]

    return df


# --------------------------------------------------
# 4️⃣ Synthetic MSRP Generator (Business Proxy)
# --------------------------------------------------
def generate_synthetic_msrp(df):

    # Base MSRP by vehicle type (approximate U.S. market ranges)
    base_msrp_map = {
        "truck": 45000,
        "pickup": 45000,
        "suv": 38000,
        "sedan": 30000,
        "hatchback": 25000,
        "coupe": 32000,
        "wagon": 28000,
        "van": 35000
    }

    df["type_lower"] = df["type"].str.lower()

    df["base_msrp"] = df["type_lower"].map(base_msrp_map)

    # If type not found → use median
    df["base_msrp"] = df["base_msrp"].fillna(32000)

    # Premium brands
    premium_brands = ["bmw", "lexus", "audi", "mercedes", "porsche"]

    df["brand_multiplier"] = np.where(
        df["manufacturer"].str.lower().isin(premium_brands),
        1.25,
        1.0
    )

    # Adjust for vehicle age (newer cars higher MSRP expectation)
    df["age_adjustment"] = 1 + (df["vehicle_age"] * 0.01)

    df["msrp_estimated"] = (
        df["base_msrp"] *
        df["brand_multiplier"] *
        df["age_adjustment"]
    )

    return df

def build_weekly_market_df(df):

    df["year_week"] = df["posting_date"].dt.to_period("W").dt.to_timestamp()

    weekly_df = (
        df.groupby("year_week")
        .agg({
            "residual_value_pct": "mean"
        })
        .reset_index()
        .sort_values("year_week")
    )

    return weekly_df


# --------------------------------------------------
# 5️⃣ Feature Engineering
# --------------------------------------------------
def add_engineered_features(df):

    df["log_price"] = np.log1p(df["price"])
    df["log_odometer"] = np.log1p(df["odometer"])

    df["price_per_mile"] = df["price"] / df["odometer"].replace(0, np.nan)

    df["age_x_log_odometer"] = df["vehicle_age"] * df["log_odometer"]

    # --------------------------
    # Residual Value %
    # --------------------------
    df["residual_value_pct"] = df["price"] / df["msrp_estimated"]

    # Sanity clipping
    df["residual_value_pct"] = df["residual_value_pct"].clip(0.05, 1.5)

    return df




# --------------------------------------------------
# 7️⃣ Full Pipeline
# --------------------------------------------------
def preprocess_pipeline():

    df = load_data()
    df = clean_data(df)
    df = filter_outliers(df)
    df = generate_synthetic_msrp(df)
    df = add_engineered_features(df)
    weekly_df = build_weekly_market_df(df)
    print("MIN DATE:", df["posting_date"].min())
    print("MAX DATE:", df["posting_date"].max())
    print("UNIQUE WEEKS:", df["posting_date"].dt.to_period("W").nunique())
    return df, weekly_df
# --------------------------------------------------
# 8️⃣ Save
# --------------------------------------------------
def save_processed(df):
    df.to_csv(PROCESSED_DATA_PATH, index=False)


if __name__ == "__main__":
    df, monthly_df = preprocess_pipeline()
    save_processed(df)
    print("\n✅ Data preprocessing completed.")
