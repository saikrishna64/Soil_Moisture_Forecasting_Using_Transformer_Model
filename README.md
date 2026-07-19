
# Soil Moisture Forecasting using Transformer Models

## Overview

This project presents a comparative study of Transformer-based deep learning models for long-horizon soil moisture forecasting. Four architectures were implemented and evaluated:

- Vanilla Transformer
- PatchTST
- iTransformer
- Cluster-Based Transformer (Proposed)

The objective is to predict future soil moisture from historical irrigation and soil moisture observations while comparing the effectiveness of different Transformer architectures.

---

## Project Objectives

- Develop deep learning models for soil moisture forecasting.
- Compare multiple Transformer architectures for time-series prediction.
- Propose a Cluster-Based Transformer framework using K-Means clustering.
- Evaluate forecasting performance using multiple regression metrics.
- Generate long-horizon autoregressive forecasts.

---

## Dataset

The dataset contains minute-level observations consisting of:

| Feature | Description |
|----------|-------------|
| Minute | Time index |
| Irrigation | Irrigation value |
| Soil Moisture | Target variable |

Dataset Statistics

- Approximately 100,000 observations
- 80% Training
- 20% Validation
- Min-Max Normalization

---

## Models Implemented

### 1. Vanilla Transformer

- Standard Transformer Encoder
- Self-attention over temporal sequence
- Baseline forecasting model

---

### 2. PatchTST

- Patch-based tokenization
- Transformer encoder
- Efficient long-sequence modeling

---

### 3. iTransformer

- Variable-wise attention
- Cross-variable representation learning
- Designed for multivariate forecasting

---

### 4. Cluster-Based Transformer (Proposed)

The proposed framework consists of:

1. Feature Extraction
2. K-Means Clustering
3. Specialist Transformer for each cluster
4. Cluster-aware forecasting

Statistical features used for clustering:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Range
- Slope
- Mean Difference
- Energy
- Last Value
- Mean Irrigation

---

## Project Workflow

```
Dataset
      │
      ▼
Preprocessing
      │
      ▼
Sequence Generation
      │
      ▼
Model Training
      │
      ▼
Validation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Evaluation
      │
      ▼
10-Day Forecast
```

---

## Evaluation Metrics

The following metrics were used for model comparison:

- RMSE
- MAE
- MAPE
- Standard R²
- Dynamic R²
- MASE

---

## Results

The proposed Cluster-Based Transformer achieved the best overall forecasting performance.

Key observations:

- Lowest RMSE
- Lowest MAE
- Lowest MAPE
- Highest Standard R²
- Outperformed the naive forecasting baseline (MASE < 1)

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- K-Means Clustering

---

## Repository Structure

```
├── Data/
├── Models/
│   ├── Vanilla Transformer
│   ├── PatchTST
│   ├── iTransformer
│   └── Cluster Transformer
├── Results/
├── Forecasts/
├── Presentation/
├── README.md
```

---

## Future Improvements

- Incorporate weather variables such as rainfall, humidity and temperature.
- Explore adaptive clustering techniques.
- Evaluate Mixture-of-Experts architectures.
- Test on larger agricultural datasets.

---
