import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from .time_series import load_cleaned_data, aggregate_daily_price, create_lag_features

def run_xgb():
    df = load_cleaned_data()
    daily = aggregate_daily_price(df)
    daily = create_lag_features(daily)

    X = daily.drop(columns=["date", "avg_price"])
    y = daily["avg_price"]

    train_size = int(len(daily) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    import matplotlib.pyplot as plt

    import pandas as pd

    importance = model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8,5))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xticks(rotation=45)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()

    plt.savefig("reports/xgb_feature_importance.png")
    plt.close()

    print("Feature importance plot saved.")




    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("XGBoost RMSE:", rmse)

if __name__ == "__main__":
    run_xgb()
