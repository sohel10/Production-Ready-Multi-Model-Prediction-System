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

---

## Business Objective

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

## Key Design Principles

- Segment-aware modeling
- Explicit schema persistence
- Separation of training and serving
- Safe feature reconstruction
- Modular architecture
- Deployment-ready containerization

---

## System Architecture

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

# 🚗 Auto Residual Value Predictor  
### Production-Grade Multi-Segment Vehicle Valuation System

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Built with](https://img.shields.io/badge/Built%20with-XGBoost%20%7C%20FastAPI%20%7C%20Docker-brightgreen)]()
[![Architecture](https://img.shields.io/badge/Architecture-Segment--Aware-blue)]()
[![Deployment](https://img.shields.io/badge/Deployment-Dockerized-success)]()
[![Status](https://img.shields.io/badge/status-production--style-success)]()

---

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

---

# 🏗️ Architecture Overview

```text
                   ┌────────────────────────────┐
                   │      Raw Vehicle Data      │
                   └─────────────┬──────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Preprocessing Layer     │
                    │  • Cleaning              │
                    │  • Feature engineering   │
                    └─────────────┬────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │ One-Hot Encoding + Schema Save │
                │ (Persisted JSON feature list)  │
                └─────────────┬──────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │ Segment-Specific XGBoost Model │
                └─────────────┬──────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │ FastAPI Inference Server       │
                │ Deterministic Reconstruction   │
                └─────────────┬──────────────────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │   Web UI    │
                          └─────────────┘


                    