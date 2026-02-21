from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import time
import os
import json
from pathlib import Path

# =========================
# App
# =========================
app = FastAPI(
    title="Auto Valuation Platform",
    version="1.0.0"
)

app.state.build_id = str(int(time.time()))

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =========================
# Config
# =========================
MODEL_DIR = Path("models")

# =========================
# Request Schema
# =========================
class VehicleInput(BaseModel):
    segment: str
    vehicle_age: float
    odometer: float
    price_per_mile: float

# =========================
# Globals
# =========================
segment_models = {}
time_model = None

# =========================
# Startup
# =========================
@app.on_event("startup")
def load_models():
    global segment_models, time_model

    segments = ["ev", "luxury", "sedan", "suv", "truck"]

    for seg in segments:
        path = MODEL_DIR / f"{seg}_segment_model.pkl"
        if path.exists():
            segment_models[seg] = joblib.load(path)

    time_model_path = MODEL_DIR / "market_time_model.pkl"
    if time_model_path.exists():
        time_model = joblib.load(time_model_path)

    print("✅ Models loaded:", list(segment_models.keys()))

# =========================
# API Endpoints
# =========================

@app.get("/health")
def health():
    return {"status": "healthy"}

import json
import numpy as np

import json
import numpy as np

@app.post("/predict")
def predict(payload: VehicleInput):

    if payload.segment not in segment_models:
        raise HTTPException(status_code=400, detail="Invalid segment")

    model = segment_models[payload.segment]

    # Load saved feature list
    feature_path = MODEL_DIR / f"{payload.segment}_features.json"

    if not feature_path.exists():
        raise HTTPException(status_code=500, detail="Feature file missing")

    with open(feature_path) as f:
        feature_columns = json.load(f)

    # Create empty frame with correct columns
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # Engineered features
    log_odometer = np.log1p(payload.odometer)
    age_x_log_odometer = payload.vehicle_age * log_odometer

    input_df["vehicle_age"] = payload.vehicle_age
    input_df["log_odometer"] = log_odometer
    input_df["age_x_log_odometer"] = age_x_log_odometer
    input_df["price_per_mile"] = payload.price_per_mile
    print("Model expects:", len(model.feature_names_in_))
    print("Input shape:", input_df.shape)
    print("Input columns:", list(input_df.columns))
# =========================
    prediction = model.predict(input_df)[0]
    prediction = np.clip(prediction, 0, 1.5)

    return {
        "segment": payload.segment,
        "predicted_residual_value_pct": round(float(prediction), 4)
    }
    
# UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "build_id": app.state.build_id
        }
    )