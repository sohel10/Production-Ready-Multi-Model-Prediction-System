import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "vehicles.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_vehicles.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FORECAST_STEPS = int(os.getenv("FORECAST_STEPS", 2))
MIN_WEEKS_VALIDATION = int(os.getenv("MIN_WEEKS_VALIDATION", 6))

SEGMENT_MODEL_CONFIG = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

TIME_MODEL_CONFIG = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8
}