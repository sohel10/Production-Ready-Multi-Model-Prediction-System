import numpy as np

def add_engineered_features(df):

    df["vehicle_age"] = 2025 - df["year"]

    df["log_price"] = np.log1p(df["price"])
    df["log_odometer"] = np.log1p(df["odometer"])

    df["price_per_mile"] = df["price"] / df["odometer"].replace(0, np.nan)

    df["is_truck"] = (df["type"] == "truck").astype(int)

    df["age_x_log_odometer"] = df["vehicle_age"] * df["log_odometer"]
    df["msrp_estimated"] = df["price"] / 0.65  # placeholder if no MSRP
    df["residual_value_pct"] = df["price"] / df["msrp_estimated"]
    return df

# --------------------------------------------------
# 6️⃣ Validation Check
# --------------------------------------------------
def validate_rv_distribution(df):

    print("\n===== RV VALIDATION =====")
    print("RV% min:", df["residual_value_pct"].min())
    print("RV% max:", df["residual_value_pct"].max())
    print("RV% mean:", df["residual_value_pct"].mean())
    print("RV% std:", df["residual_value_pct"].std())

    # ---------------------------
    # 📈 Monthly Market Index
    # ---------------------------

    df["year_month"] = df["posting_date"].dt.to_period("M")

    monthly_avg_price = (
        df.groupby("year_month")["price"]
        .mean()
        .rename("monthly_avg_price")
    )

    df = df.merge(
        monthly_avg_price,
        on="year_month",
        how="left"
    )

    # Normalize market index (base = 1)
    df["market_index"] = df["monthly_avg_price"] / df["monthly_avg_price"].mean()

    # Inflation-adjusted price
    df["price_adj"] = df["price"] / df["market_index"]

    # Log adjusted price
    df["log_price_adj"] = np.log1p(df["price_adj"])
    return df