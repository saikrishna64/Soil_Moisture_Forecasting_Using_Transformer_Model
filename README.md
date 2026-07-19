
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
├── Cluster_Based_Transformer/
│   ├──Cluster_based_Transformers_1.ipynb
│   ├──cluster_evaluation.png
│   ├──cluster_forecast_10days.png
│   ├──cluster_selection.png
│   ├──cluster_training_curves.png
│   ├──cluster_visualisation.png
│   └──forecast_14400_cluster_transformer.csv
├── PatchTST/
│   ├──best_patchtst_sm_model.pt
│   ├──forecast_14400_patchtst.xlsx
│   ├──patchtst_evaluation.png
│   ├──patchtst_forecast_10days.png
│   ├──patchtst_training_curve.png
│   ├──plot1_actual_vs_predicted.png
│   ├──plot2_horizon_error.png
│   ├──plot3_r2_mase.png
│   ├──plot4_residuals.png
│   └──plot5_scatter.png
├── Vanilla_Transformer/
│   ├──best_vanilla_transformer.pt
│   ├──evaluation_metrics.png
│   ├──forecast_10days.png
│   ├──forecast_14400_vanilla_transformer.csv
│   └──training_curve.png
├── iTransformers/
│   ├──best_itransformer.pt
│   ├──forecast_14400_itransformer.csv
│   ├──itransformer_evaluation.png
│   ├──itransformer_forecast_10days.png
│   └──itransformer_training_curve.png
├── Presentation/
│   └──itransformer_training_curve.png
├── Final_Dataset.xlsx
├── Supervised_Transformer_models.ipynb
└── README.md
```

---

## Future Improvements

- Incorporate weather variables such as rainfall, humidity and temperature.
- Explore adaptive clustering techniques.
- Evaluate Mixture-of-Experts architectures.
- Test on larger agricultural datasets.

---
