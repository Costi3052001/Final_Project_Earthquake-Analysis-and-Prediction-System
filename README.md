# Earthquake Analysis and Prediction System

A machine learning pipeline for earthquake risk classification, magnitude prediction, and real-time monitoring using USGS data.

## Overview

This project fetches seismic data from the USGS API, trains machine learning and deep learning models for risk classification and magnitude prediction, and provides real-time earthquake monitoring with automated response recommendations.

## Features

- Automated data collection from USGS (2 years of M ≥ 2.5 earthquakes)
- Risk classification (Low/Medium/High) using ML and deep learning models
- Magnitude prediction using regression models
- LSTM-based time series forecasting for daily earthquake counts
- Real-time monitoring with risk assessment and response recommendations

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow requests
```

## Data Source

- **API**: USGS Earthquake Hazards Program
- **Endpoint**: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- **Format**: GeoJSON

## Models

### Classification (Risk Level)

| Model | Description |
|-------|-------------|
| Random Forest | 150 estimators, balanced class weights |
| Logistic Regression | Balanced weights |
| KNN | k=7 neighbors |
| DNN | 3-layer network with dropout |

### Regression (Magnitude)

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Random Forest | 200 estimators |
| Gradient Boosting | 200 estimators |
| DNN | 4-layer network with dropout |

### Time Series

- **LSTM**: Predicts daily earthquake counts using 7-day sequences (can be edited for any number or region)

## Model Evaluation

### Cross-Validation Comparison

Models are compared using:

- Paired t-tests for statistical significance between models

### Metrics

- **Classification**: Accuracy, F1-score (weighted/macro), confusion matrix
- **Regression**: RMSE, MAE, R²

## Digital Twin System

The real-time monitoring system:
1. Fetches latest earthquakes from USGS feeds
2. Predicts risk level and magnitude
3. Generates response recommendations

**Alert Levels:**
- **CRITICAL**: High risk / M ≥ 6.0 → Full emergency response
- **WARNING**: Medium risk / M ≥ 4.5 → Increased monitoring
- **NORMAL**: Low risk → Routine monitoring


## References

- USGS Earthquake Hazards Program: https://earthquake.usgs.gov/

