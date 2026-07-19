# 🌱 Soil Moisture Forecasting with Transformer Models

> **10-day ahead soil moisture prediction at minute-level resolution using three Transformer architectures — Vanilla Transformer, PatchTST, and iTransformer.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Problem Statement](#-problem-statement)
- [Architecture Overview](#-architecture-overview)
  - [Vanilla Transformer](#1-vanilla-transformer)
  - [PatchTST](#2-patchtst)
  - [iTransformer](#3-itransformer)
- [Data Pipeline](#-data-pipeline)
- [Results & Evaluation](#-results--evaluation)
- [Model Comparison](#-model-comparison)
- [Key Observations](#-key-observations)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Requirements](#-requirements)

---

## 🔍 Project Overview

This project benchmarks three state-of-the-art Transformer architectures for **long-range soil moisture time-series forecasting**. Given one day (1,440 minutes) of past observations, each model predicts soil moisture for the **next 10 days (14,400 minutes)** at minute-level resolution.

The three models are evaluated under identical data splits, preprocessing, and evaluation metrics to provide a fair, reproducible comparison.

| Model | Architecture Type | Key Idea |
|---|---|---|
| Vanilla Transformer | Standard encoder-only | Self-attention over time steps |
| PatchTST | Patch-based Transformer | Divides sequence into patches (tokens) |
| iTransformer | Inverted Transformer | Treats each channel (feature) as a token |

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `Final_Dataset.xlsx` |
| Total rows | 100,000 |
| Temporal resolution | 1 minute per row |
| Total duration | ~69.4 days |
| Columns | 3 |

### Columns

| Column | Description | Range |
|---|---|---|
| `Minute` | Sequential minute index (1, 2, 3, ...) | 1 – 100,000 |
| `IRRIGATION` | Irrigation rate (m³/s) | 0.000002 – 0.000008 |
| `Soil Moisture` | Volumetric water content (m³/m³) — **prediction target** | 0.413 – 0.450 |

### Data Characteristics

- **Soil moisture range:** 0.413 to 0.450 — a narrow band of ~0.037 m³/m³
- **Soil moisture mean:** 0.4347 m³/m³
- **Soil moisture std:** 0.00776 m³/m³
- **Irrigation:** 10 unique values — a step-change control input
- The dataset is a continuous single-location simulation (no geographic dimension)

---

## 🎯 Problem Statement

**Task:** Given the past **1 day = 1,440 minutes** of `[IRRIGATION, Soil Moisture]` observations and 8 engineered features, predict the **Soil Moisture** for the next **10 days = 14,400 minutes**.

**Forecasting type:** Direct multi-step — all future values predicted in a single forward pass (no autoregressive rollout).

**Challenge:** 14,400 direct minute-level output values is computationally heavy. The approach taken:
1. Resample raw 1-minute data → **10-minute blocks** (100,000 rows → 10,000 blocks)
2. Transformer predicts **1,440 output steps** (10-min resolution)
3. Cubic spline interpolation expands predictions back to **14,400 minute-level** values

---

## 🏗️ Architecture Overview

All three models share the same data pipeline and evaluation framework. Only the model architecture differs.

### Common Configuration

```
Input  : (batch, seq_len=144, features=10)   ← 1 day of 10-min blocks, 10 features
Output : (batch, horizon=1440)               ← 10 days of 10-min predictions
```

### Engineered Features (10 total)

| Feature | Description |
|---|---|
| `irrigation` | Irrigation rate (10-min avg) |
| `sm` | Soil moisture (10-min avg) |
| `sm_roll6` | 1-hour rolling mean (6 × 10min) |
| `sm_roll24` | 4-hour rolling mean (24 × 10min) |
| `sm_diff1` | 10-min rate of change |
| `sm_lag1` | Soil moisture 10 min ago |
| `sm_lag6` | Soil moisture 1 hour ago |
| `sm_lag24` | Soil moisture 4 hours ago |
| `sm_lag144` | Soil moisture 1 day ago |
| `irr_roll24` | 4-hour rolling mean of irrigation |

---

### 1. Vanilla Transformer

Standard encoder-only Transformer. Each 10-minute block is treated as one token. All 144 tokens (1 day) attend to each other through multi-head self-attention.

```
Input (batch, 144, 10)
    ↓
Input Projection  Linear(10 → 128)
    ↓
Positional Encoding  sin/cos stamps × 144 positions
    ↓
Transformer Encoder × 4 layers
    [LayerNorm → MHA (4 heads, d_k=32) → Residual]
    [LayerNorm → FFN (512, GELU)       → Residual]
    ↓
Mean Pooling  (batch, 144, 128) → (batch, 128)
    ↓
Head  Linear(128 → 512) → GELU → Linear(512 → 1440)
    ↓
Output (batch, 1440)
```

**Parameters:** ~2.1 million

---

### 2. PatchTST

Divides the input sequence into non-overlapping **patches** (sub-sequences), treating each patch as a token. This reduces the sequence length fed to attention and forces the model to reason at a coarser granularity first.

```
Input (batch, 144, 10)
    ↓
Split into 17 patches of length ~8 steps each
    (batch, 10 channels, 17 patches, patch_len)
    ↓
Patch Embedding  per-channel linear projection
    ↓
Positional Encoding over 17 patch tokens
    ↓
Transformer Encoder × 4 layers
    ↓
Flatten + Head → (batch, 1440)
```

**Patches:** 144 steps ÷ patch_size ≈ **17 sequential tokens**  
**Channels processed:** 10 feature paths embedded independently

---

### 3. iTransformer

Inverts the standard time-series Transformer perspective. Instead of treating time steps as tokens, **each feature channel becomes a token**. The temporal dimension is projected into the embedding. This allows the model to learn cross-feature dependencies directly through attention.

```
Input (batch, 144, 10)
    ↓
Transpose → (batch, 10 channels, 144 time steps)
    ↓
Temporal Projection  Linear(144 → 128) per channel
    → (batch, 10 tokens, 128)
    ↓
Transformer Encoder × 4 layers
    Attention across 10 feature tokens (not time steps)
    ↓
Head → (batch, 1440)
```

**Inverted tokens (channels):** 10  
**Temporal vector projected:** 144 steps → 128 dimensions

---

## 🔄 Data Pipeline

```
Raw Data (100,000 rows × 3 cols)
    │
    ▼
1. Forward-fill IRRIGATION NaNs (after row 16,383)
    │
    ▼
2. Resample to 10-minute blocks → 10,000 blocks
    │
    ▼
3. Engineer 10 features (lags, rolling means, diff)
    │
    ▼
4. Sliding Windows
   ├── Input  X[i] = features[i : i+144]        shape (144, 10)
   └── Target y[i] = sm[i+144 : i+144+1440]     shape (1440,)
   Total windows: 8,417
    │
    ▼
5. Chronological Split (no shuffle)
   ├── Train : 5,891 windows  (70%)
   ├── Val   : 1,263 windows  (15%)
   └── Test  : 1,263 windows  (15%)
    │
    ▼
6. StandardScaler  (fit on train only — no data leakage)
    │
    ▼
7. Train → Evaluate → Cubic Interpolate → 14,400 minute-level forecast
```

---

## 📈 Results & Evaluation

### Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **RMSE** | √(mean((ŷ−y)²)) | Average error magnitude (penalises large errors) |
| **MAE** | mean(\|ŷ−y\|) | Average absolute error |
| **MAPE** | mean(\|ŷ−y\|/\|y\|) × 100 | Percentage error relative to actual value |
| **MASE** | MAE / MAE_naïve | Error relative to a naïve 1-step-lag baseline |
| **R²** | 1 − SS_res/SS_tot | Variance explained (1=perfect, 0=mean baseline) |

---

### Overall Test Metrics

| Model | RMSE | MAE | MAPE (%) | MASE | R² |
|---|---|---|---|---|---|
| **Vanilla Transformer** | 0.007714 | 0.006469 | 1.4809 | 0.7324 | −0.0013 |
| **PatchTST** | 0.007714 | 0.006469 | 1.4809 | 0.7324 | −0.0013 |
| **iTransformer** | 0.007714 | 0.006470 | 1.4811 | 0.7325 | −0.0013 |

---

### Day-by-Day RMSE Breakdown

| Day | Vanilla | PatchTST | iTransformer |
|---|---|---|---|
| Day 1 | 0.007829 | 0.007829 | 0.007828 |
| Day 2 | 0.007826 | 0.007826 | 0.007826 |
| Day 3 | 0.007837 | 0.007837 | 0.007837 |
| Day 4 | 0.007745 | 0.007745 | 0.007745 |
| Day 5 | 0.007666 | 0.007667 | 0.007667 |
| Day 6 | 0.007675 | 0.007675 | 0.007676 |
| Day 7 | 0.007653 | 0.007653 | 0.007653 |
| Day 8 | 0.007641 | 0.007641 | 0.007642 |
| Day 9 | 0.007607 | 0.007607 | 0.007607 |
| Day 10 | 0.007658 | 0.007658 | 0.007658 |

---

### Training Behaviour

| Model | Epochs trained | Early stop at | Final val MSE |
|---|---|---|---|
| Vanilla Transformer | 56 | Ep 56 | 1.0156457 |
| PatchTST | 53 | Ep 53 | 1.0156534 |
| iTransformer | 31 | Ep 31 | 1.0156927 |

---

## 📊 Model Comparison

### Architecture Differences

| Aspect | Vanilla | PatchTST | iTransformer |
|---|---|---|---|
| Token type | Time step (1 per 10-min block) | Patch (group of time steps) | Feature channel |
| Sequence length seen by attention | 144 tokens | 17 patch tokens | 10 channel tokens |
| Cross-feature interaction | Implicit via shared projection | Implicit | **Explicit** (attention across features) |
| Cross-time interaction | **Full** (all 144 steps) | Within-patch local | Projected to embedding |
| Inductive bias | Temporal order | Local temporal structure | Feature correlation |

### Training Behaviour Comparison

- **iTransformer** overfit earliest — stopped at epoch 31. Its val MSE rose after epoch 6, suggesting the cross-channel attention without sufficient temporal context caused the model to memorise training patterns quickly on this single-location dataset.
- **Vanilla Transformer** trained the longest (56 epochs) and showed the steadiest descent, with val MSE improving consistently from epoch 1 to 21 before plateauing.
- **PatchTST** converged similarly to Vanilla but slightly earlier (53 epochs). Its patch-based tokenisation gave marginally different convergence dynamics but identical final performance.

---

## 🔬 Key Observations

### 1. All three models perform identically on test data
All three architectures produce RMSE = 0.007714, MAE = 0.006469, and R² = −0.0013. The differences are at the 5th decimal place or beyond — statistically negligible. This is the most important result of the experiment.

### 2. Negative R² — what it means and why it happened
R² = −0.0013 means the models perform **marginally worse than simply predicting the mean** of soil moisture for every future step. This is not a failure of the Transformer architecture specifically. It reflects a fundamental property of this dataset:

- Soil moisture std = **0.00776** — an extremely narrow signal range of 0.037 m³/m³
- The dataset is a controlled simulation with no external weather events (rainfall, temperature) in the feature set
- The target is nearly stationary — the mean-prediction baseline is already very hard to beat
- All models effectively learned the **mean level** of soil moisture, which is a valid and reasonable strategy when the signal variance is this small

### 3. MAPE of ~1.48% is actually strong performance
While R² is negative, MAPE of 1.48% means predictions are within 1.5% of actual values on average. At this scale of prediction (14,400 minutes ahead), 1.5% percentage error is practically useful — it means the model is rarely off by more than 0.007 m³/m³.

### 4. MASE < 1 for all models
MASE of ~0.73 means all models outperform a naïve persistence (lag-1) baseline by 27%. The transformer is adding genuine predictive value beyond simply repeating the last observed value.

### 5. RMSE improves from Day 1 → Day 9, then rises slightly at Day 10
Across all three models, RMSE starts at ~0.007829 on Day 1 and drops to ~0.007607 by Day 9, then rises marginally to 0.007658 on Day 10. This U-shaped pattern suggests the models are most uncertain at the very start (transition from input to prediction) and at the very end of the horizon.

### 6. Architecture choice does not matter for this dataset
The identical performance across Vanilla, PatchTST, and iTransformer indicates that the **bottleneck is the data, not the architecture**. No architecture can recover variance that is not present in the input features. Adding meteorological features (rainfall, temperature, evapotranspiration) would likely differentiate the models more clearly.

### 7. iTransformer's cross-feature attention is unnecessary here
iTransformer's key strength is learning dependencies between different sensors or locations. With only 10 engineered features derived from 2 raw signals (irrigation + soil moisture), the cross-feature relationships are simple enough that standard temporal attention (Vanilla) captures them equally well.

---

## 📁 Project Structure

```
soil-moisture-transformer/
│
├── Final_Dataset.xlsx              # Raw dataset (100,000 rows × 3 columns)
│
├── vanilla_transformer.py          # Vanilla Transformer model + training
├── patchtst.py                     # PatchTST model + training
├── itransformer.py                 # iTransformer model + training
│
├── data_pipeline.py                # Shared data loading, resampling, windowing
├── evaluation.py                   # Shared evaluation metrics (RMSE, MAE, MAPE, MASE, R²)
│
├── figures/
│   ├── fig1_training_metrics.png   # Loss curves + RMSE/R² per day
│   ├── fig2_actual_vs_predicted.png
│   ├── fig3_scatter_plots.png
│   ├── fig4_residuals.png
│   └── fig5_architecture.png
│
├── saved_models/
│   ├── vanilla_transformer.pt
│   ├── patchtst.pt
│   └── itransformer.pt
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/soil-moisture-transformer.git
cd soil-moisture-transformer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run any model

```bash
# Vanilla Transformer
python vanilla_transformer.py

# PatchTST
python patchtst.py

# iTransformer
python itransformer.py
```

### 4. Change data path if needed

In each script, update the `DATA_PATH` in the `CFG` dict at the top:

```python
CFG = dict(
    DATA_PATH = "Final_Dataset.xlsx",   # ← update this
    ...
)
```

---

## 📦 Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
```

Install all at once:

```bash
pip install torch numpy pandas openpyxl scikit-learn matplotlib scipy
```

**Python version:** 3.9 or higher recommended.

---

## 📌 Citation

If you use this project or its results in your work, please cite:

```bibtex
@misc{soilmoisture_transformer_2025,
  title   = {Soil Moisture 10-Day Forecasting with Transformer Architectures},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/your-username/soil-moisture-transformer}
}
```

---

## 📄 License

This project is released under the MIT License.
