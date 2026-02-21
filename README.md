# 🚗 Auto Residual Value Predictor  
### Production-Style Multi-Segment Vehicle Valuation System

- Multi-segment XGBoost modeling (SUV, Sedan, Truck, Luxury, EV)
- Feature schema persistence for safe inference
- Production-style FastAPI serving
- Dockerized deployment-ready architecture
- Modular training vs serving separation
- Designed for scalable, segment-aware pricing systems

---

## Overview

This repository implements a **production-style, multi-segment residual value prediction system** designed to estimate used vehicle residual value percentages (RV%) in a scalable and deployment-ready manner.

Instead of training a single monolithic model, this system:

- Trains segment-specific models  
- Persists exact feature schemas  
- Reconstructs inference inputs safely  
- Separates training from serving  
- Containerizes the API layer  

This architecture reflects how **real-world valuation systems** are deployed in automotive analytics environments.
## 🧰 Tech Stack

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)](https://spark.apache.org/docs/latest/api/python/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF)](https://github.com/features/actions)
[![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-FCC624)](https://ubuntu.com/)
[![Parquet](https://img.shields.io/badge/Storage-Parquet-4B8BBE)](https://parquet.apache.org/)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Built with](https://img.shields.io/badge/Built%20with-XGBoost%20%7C%20FastAPI%20%7C%20Docker-brightgreen)]()
[![Architecture](https://img.shields.io/badge/Architecture-Segment--Aware-blue)]()
[![Deployment](https://img.shields.io/badge/Deployment-Dockerized-success)]()
[![Status](https://img.shields.io/badge/status-production--style-success)]()

Predict **Residual Value Percentage (RV%)** of used vehicles using:

- Vehicle Age  
- Odometer  
- Price Per Mile  
- Manufacturer  
- State  
- Transmission  
- Drive Type  
- Fuel Type  

Models are trained independently for:

- SUV  
- Sedan  
- Truck  
- Luxury  
- EV  

Segment-specific modeling improves performance and reduces cross-category bias.

---

## Why This Architecture Works

Instead of using a single global model, this system:

- Decomposes valuation by segment
- Persists training schema for inference safety
- Reconstructs feature matrices deterministically
- Ensures production-safe prediction pipelines

The inference server **never guesses feature order**.

Feature reconstruction is fully deterministic.

---
## 🎬 Live Demo

<p align="center">
  <img src="Animation.gif" alt="Clinical RAG Assistant Demo" width="850"/>
</p>

> This repository focuses on **data ingestion, chunking, and embeddings**.
## Key Design Principles

- Segment-aware modeling
- Explicit schema persistence
- Separation of training and serving
- Safe feature reconstruction
- Modular architecture
- Deployment-ready containerization

---

## System Architecture



# 🚗 Auto Residual Value Predictor  
### Production-Grade Multi-Segment Vehicle Valuation System


## 🚀 What This System Demonstrates

- Multi-segment ML modeling (SUV, Sedan, Truck, Luxury, EV)
- Deterministic feature schema persistence
- Production-safe inference reconstruction
- Modular training vs serving separation
- Containerized API deployment
- Deployment-ready architecture
- Segment-aware pricing intelligence

This repository reflects how **real-world automotive valuation systems are engineered and deployed**.

---

## 🎯 Business Objective

Predict **Residual Value Percentage (RV%)** for used vehicles using:

- Vehicle Age
- Odometer
- Price Per Mile
- Manufacturer
- State
- Transmission
- Drive Type
- Fuel Type

Instead of a single global model, this system trains **segment-specific models** to reduce cross-category bias and improve predictive accuracy.

---

## 🧠 Why This Architecture Works

Naïve approach:
- Train one global model
- Hope inference matches training schema
- Risk feature mismatch in production

This system instead:

- Trains segment-specific models
- Persists feature schemas as JSON
- Reconstructs inference input deterministically
- Guarantees feature shape consistency
- Separates training from serving

The inference server **never guesses feature order**.


## 📊 Model Performance & Interpretability
# 📊 Model Evaluation & Diagnostics

This project solves a **regression problem** (Residual Value % prediction), therefore performance is evaluated using regression metrics and visual diagnostics — not ROC curves.

---

## 📈 Model Performance (Regression Metrics)

The model is evaluated using:

- RMSE (Root Mean Squared Error)
- Segment-level performance comparison
- Out-of-sample validation

![Model Performance](reports/figures/model_performance.png)

> Lower RMSE indicates better predictive accuracy.

---

## 📉 Residual Value Distribution

Distribution of predicted residual values across vehicle segments:

![Residual Value Distribution](reports/figures/rv_distribution.png)

This visualization helps assess prediction spread and detect skewness or instability.

---

## 📊 Weekly Market Forecast (Time-Series Extension)

Short-term forecast of market-level residual value trends using time-series modeling.

![Market Weekly Forecast](reports/figures/market_weekly_forecast.png)

This provides forward-looking insight into depreciation dynamics.

---

## 🚗 Segment-Level Depreciation Curves

### SUV Depreciation

![SUV Depreciation](reports/figures/suv_depreciation.png)

### Sedan Depreciation

![Sedan Depreciation](reports/figures/sedan_depreciation.png)

### Truck Depreciation

![Truck Depreciation](reports/figures/truck_depreciation.png)

### Luxury Depreciation

![Luxury Depreciation](reports/figures/luxury_depreciation.png)

### EV Depreciation

![EV Depreciation](reports/figures/ev_depreciation.png)

---

## 🔎 Feature Importance (XGBoost Gain-Based)

Top features driving regression predictions:

![Feature Importance](reports/figures/xgb_feature_importance.png)

> Feature importance reflects global model contribution and does not imply causality.

---

# ⚙️ MLOps & Production Engineering

This project follows **production-oriented ML system design** to ensure reproducibility, reliability, and safe deployment.

---

## 🔌 Inference API (FastAPI)

A production-ready inference service is implemented using **FastAPI**, exposing model predictions via REST endpoints.

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Predict residual value percentage for a vehicle |

The API enforces:

- Strict input schema validation  
- Deterministic feature reconstruction  
- Alignment with training feature schema  
- Stateless inference  
- Safe error handling  

---

## 🧪 CI/CD (GitHub Actions)

This repository uses **GitHub Actions** to automate quality checks.

### CI Pipeline Includes:

- Python environment setup  
- Dependency installation  
- Model import validation  
- API startup verification  
- Inference sanity checks  

This ensures:

- Broken code is caught early  
- API behavior remains stable  
- Model inference does not silently regress  

### High-Level Flow

```text
Raw Data
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
One-Hot Encoding
   ↓
Feature Schema Saved (JSON)
   ↓
Segment-Specific XGBoost Training
   ↓
Model Artifacts Saved (.pkl)
   ↓
FastAPI Inference Layer
   ↓
Web UI

## Project Structure
auto-valuation/
│
├── src/
│   ├── preprocessing.py
│   ├── residual_model.py
│   ├── train_time_model.py
│   ├── config.py
│   └── api.py
│
├── models/
│   ├── *_segment_model.pkl
│   ├── *_features.json
│   └── market_time_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── reports/
├── Dockerfile
├── requirements.txt
└── README.md

### 🚀 Model Serving

The FastAPI service can be launched locally or deployed in a containerized environment.

```bash
uvicorn app.main:app --reload

uvicorn app.main:app --reload
