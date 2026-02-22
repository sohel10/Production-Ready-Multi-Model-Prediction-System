import pandas as pd
import numpy as np
from src.config import MIN_WEEKS_VALIDATION
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import joblib
import shap
from sklearn.metrics import mean_squared_error
from .preprocessing import preprocess_pipeline
from .train_time_model import train_time_model, forecast_future
import argparse
from .config import FORECAST_STEPS, MIN_WEEKS_VALIDATION
from .config import SEGMENT_MODEL_CONFIG
from .config import TIME_MODEL_CONFIG

# --------------------------------------------------
# 📁 Report Directories
# --------------------------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures")
MODEL_DIR = "models"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


# --------------------------------------------------
# 1️⃣ Validation
# --------------------------------------------------

def validate_dataset(df):

    print("\n===== DATA VALIDATION =====")
    print("Total rows:", len(df))

    if "residual_value_pct" not in df.columns:
        raise ValueError("residual_value_pct column missing")

    print("Missing RV%:", df["residual_value_pct"].isna().sum())
    print("RV% min:", df["residual_value_pct"].min())
    print("RV% max:", df["residual_value_pct"].max())


# --------------------------------------------------
# 2️⃣ Segment Builder
# --------------------------------------------------

def get_segments(df):

    df["type"] = df["type"].str.lower()
    df["manufacturer"] = df["manufacturer"].str.lower()
    df["fuel"] = df["fuel"].str.lower()

    return {
        "truck": df[df["type"] == "truck"],
        "suv": df[df["type"] == "suv"],
        "sedan": df[df["type"] == "sedan"],
        "luxury": df[df["manufacturer"].isin(["bmw", "lexus", "audi"])],
        "ev": df[df["fuel"] == "electric"]
    }


# --------------------------------------------------
# 3️⃣ Model Training
# --------------------------------------------------

def train_segment_model(segment_df, segment_name):

    if len(segment_df) < 500:
        print(f"Skipping {segment_name} (too small)")
        return None

    segment_df = segment_df.sort_values("posting_date")
    split_date = segment_df["posting_date"].quantile(0.8)

    train_df = segment_df[segment_df["posting_date"] < split_date]
    test_df = segment_df[segment_df["posting_date"] >= split_date]

    if len(train_df) == 0 or len(test_df) == 0:
        print(f"Skipping {segment_name} (empty split)")
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

    X_train = pd.get_dummies(train_df[feature_cols + categorical_cols], drop_first=True)
    X_test = pd.get_dummies(test_df[feature_cols + categorical_cols], drop_first=True)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    import json

    feature_path = os.path.join(MODEL_DIR, f"{segment_name}_features.json")
    with open(feature_path, "w") as f:
        json.dump(X_train.columns.tolist(), f)

    y_train = train_df["residual_value_pct"]
    y_test = test_df["residual_value_pct"]

    model = xgb.XGBRegressor(
    **SEGMENT_MODEL_CONFIG,
    random_state=42,
    n_jobs=-1
)

    model.fit(X_train, y_train)
    

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{segment_name.upper()} RMSE (RV%): {rmse:.4f}")
    print(f"Saved {len(X_train.columns)} features for {segment_name}")

    return model, rmse, X_test


# --------------------------------------------------
# 4️⃣ Depreciation Simulator
# --------------------------------------------------

def simulate_depreciation(model, base_vehicle, months=36):

    results = []

    for m in range(months + 1):

        temp = base_vehicle.copy()

        temp["vehicle_age"] += m / 12
        temp["odometer"] += 1000 * m
        temp["log_odometer"] = np.log1p(temp["odometer"])
        temp["age_x_log_odometer"] = temp["vehicle_age"] * temp["log_odometer"]

        X = pd.get_dummies(pd.DataFrame([temp]), drop_first=True)
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

        pred_rv = model.predict(X)[0]
        results.append((m, pred_rv))

    return pd.DataFrame(results, columns=["month", "predicted_rv"])


# --------------------------------------------------
# 5️⃣ SHAP Plot
# --------------------------------------------------

def save_shap_plot(model, X_sample, segment_name):

    print(f"Saving SHAP plot for {segment_name.upper()}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    save_path = os.path.join(FIG_DIR, f"{segment_name}_shap_summary.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


# --------------------------------------------------
# 6️⃣ Visualization
# --------------------------------------------------

def save_segment_plot(name, curve):

    plt.figure(figsize=(8, 6))
    plt.plot(curve["month"], curve["predicted_rv"], linewidth=3)

    plt.title(f"{name.upper()} – 36 Month Residual Forecast")
    plt.xlabel("Months")
    plt.ylabel("Residual Value (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}_depreciation.png"), dpi=300)
    plt.close()


def save_dashboard(curves_dict):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (name, curve) in zip(axes, curves_dict.items()):
        ax.plot(curve["month"], curve["predicted_rv"])
        ax.set_title(name.upper())
        ax.set_xlabel("Months")
        ax.set_ylabel("RV %")
        ax.grid(True, alpha=0.3)

    for i in range(len(curves_dict), 6):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "executive_dashboard.png"), dpi=300)
    plt.close()


def save_performance_chart(rmse_dict):

    names = list(rmse_dict.keys())
    values = list(rmse_dict.values())

    plt.figure(figsize=(8, 6))
    plt.bar(names, values)

    plt.title("Segment Model RMSE")
    plt.ylabel("RMSE")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_performance.png"), dpi=300)
    plt.close()


def save_rv_distribution(df):

    plt.figure(figsize=(8, 6))
    plt.hist(df["residual_value_pct"], bins=50)

    plt.title("Residual Value Distribution")
    plt.xlabel("Residual Value %")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rv_distribution.png"), dpi=300)
    plt.close()


def save_time_forecast_plot(weekly_df, short_forecast):

    plt.figure(figsize=(10, 6))

    plt.plot(
        weekly_df["year_week"],
        weekly_df["residual_value_pct"],
        label="Historical",
        linewidth=2
    )

    last_date = weekly_df["year_week"].max()

    short_dates = pd.date_range(
        last_date,
        periods=3,
        freq="W"
    )[1:]

    plt.plot(short_dates, short_forecast, label="2W Forecast")

    plt.legend()
    plt.title("Weekly Market Residual Value Forecast")
    plt.ylabel("Residual Value %")
    plt.tight_layout()

    plt.savefig(os.path.join(FIG_DIR, "market_weekly_forecast.png"), dpi=300)
    plt.close()


# --------------------------------------------------
# 7️⃣ Main Runner
# --------------------------------------------------

def run_residual_system():

    # -------------------------------
    # CLI Override Layer
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_steps", type=int, default=FORECAST_STEPS)
    parser.add_argument("--min_weeks", type=int, default=MIN_WEEKS_VALIDATION)
    args = parser.parse_args()

    forecast_steps = args.forecast_steps
    min_weeks_validation = args.min_weeks

    logger.info("Starting Residual System Pipeline")

    # -------------------------------
    # 1️⃣ Data + Segment Models
    # -------------------------------
    df, weekly_df = preprocess_pipeline()
    validate_dataset(df)

    segments = get_segments(df)

    trained_models = {}
    rmse_results = {}
    curves = {}

    for name, segment_df in segments.items():

        output = train_segment_model(segment_df, name)

        if output:
            model, rmse, X_test = output

            trained_models[name] = model
            rmse_results[name] = rmse

            joblib.dump(
                model,
                os.path.join(MODEL_DIR, f"{name}_segment_model.pkl")
            )

            sample_vehicle = segment_df.iloc[0].to_dict()
            curve = simulate_depreciation(model, sample_vehicle)

            curves[name] = curve
            save_segment_plot(name, curve)

            if name == "suv" and len(X_test) > 500:
                save_shap_plot(
                    model,
                    X_test.sample(500, random_state=42),
                    name
                )

    save_dashboard(curves)
    save_performance_chart(rmse_results)
    save_rv_distribution(df)

    logger.info("Segment training completed")
    logger.info("Figures saved inside reports/figures/")

    # -------------------------------
    # 2️⃣ Time Series Forecast
    # -------------------------------
    if len(weekly_df) < min_weeks_validation:
        logger.warning("Not enough weekly history for forecasting.")
        return trained_models

    logger.info("Training Weekly Market Time Model...")

    time_model = train_time_model(weekly_df)

    joblib.dump(
        time_model,
        os.path.join(MODEL_DIR, "market_time_model.pkl")
    )

    short_term_forecast = forecast_future(
        time_model,
        weekly_df,
        forecast_steps
    )

    save_time_forecast_plot(
        weekly_df,
        short_term_forecast
    )

    logger.info(f"Short-term ({forecast_steps} weeks) forecast generated")

    return trained_models


if __name__ == "__main__":
    run_residual_system()