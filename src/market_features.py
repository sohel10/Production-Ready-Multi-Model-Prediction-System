def add_market_index(df):

    df["year_month"] = df["posting_date"].dt.to_period("M")

    monthly_index = df.groupby("year_month")["price"].mean()

    df = df.merge(
        monthly_index.rename("market_index"),
        on="year_month"
    )

    return df
