# 🎵 Lab Work 2 — Spotify Genre Classification & Model Comparison

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that classifies **Spotify songs into playlist genres** using audio features. Multiple ML models are trained, tuned with hyperparameter search, and compared across Accuracy, F1-score, ROC-AUC and LogLoss metrics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models Compared](#models-compared)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [References](#references)

---

## Overview

The goal of this project is to predict the **playlist genre** of a Spotify track (pop, rap, rock, latin, r&b, edm) from its audio features alone — such as danceability, energy, tempo, speechiness, and more.

The full pipeline covers:

> **EDA → Cleaning → SMOTE Balancing → Scaling → PCA → Model Training → Hyperparameter Tuning → Evaluation → Comparison**

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **File** | `spotify_songs.csv` |
| **Source** | Spotify / TidyTuesday open dataset |
| **Rows** | 32,833 songs |
| **Columns** | 23 features |
| **Target** | `playlist_genre` (6 classes) |
| **Genres** | pop, rap, rock, latin, r&b, edm |
| **Subgenres** | 24 (e.g. dance pop, trap, album rock, reggaeton, neo soul, big room…) |

### Audio Feature Columns Used for ML

| Feature | Description |
|---------|-------------|
| `danceability` | How suitable a track is for dancing (0.0–1.0) |
| `energy` | Intensity and activity level (0.0–1.0) |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words |
| `acousticness` | Whether the track is acoustic |
| `instrumentalness` | Likelihood of no vocals |
| `liveness` | Presence of live audience |
| `valence` | Musical positiveness (0.0–1.0) |
| `tempo` | Estimated BPM |
| `duration_ms` | Track length in milliseconds |
| `track_popularity` | Spotify popularity score (0–100) |

---

## Project Structure

```
📦 lab-work-2/
 ┣ 📓 Lab_work_2.ipynb          # Main notebook — full pipeline
 ┣ 📊 spotify_songs.csv         # Dataset (32,833 songs × 23 features)
 ┣ 📦 best_<ModelName>.pkl      # Saved best model (generated on run)
 ┗ 📄 README.md                 # This file
```

---

## Pipeline

```
1. Load Data          →  pd.read_csv()
2. EDA                →  .info(), .describe(), .isnull().sum()
3. Visualisation      →  Genre dist., KDE, heatmap, Plotly charts
4. Cleaning           →  dropna(), LabelEncoder on target
5. Train/Test Split   →  80/20, stratified
6. SMOTE Balancing    →  Oversample minority genres
7. Feature Scaling    →  StandardScaler
8. PCA                →  Retain 95% variance
9. Model Training     →  RandomizedSearchCV (5-iter, 3-fold CV)
10. Evaluation        →  Accuracy, F1, ROC-AUC, LogLoss
11. Comparison        →  Bar chart, Radar plot, ROC curves, Confusion matrices
12. Save Best Model   →  joblib.dump()
```

---

## Models Compared

| Model | Hyperparameters Tuned |
|-------|-----------------------|
| **Logistic Regression** | C, solver |
| **K-Nearest Neighbours** | n_neighbors, weights |
| **Support Vector Machine** | C, kernel |
| **Decision Tree** | max_depth, min_samples_split |
| **Random Forest** | n_estimators, max_depth |
| **Gradient Boosting** | n_estimators, learning_rate |
| **MLP Neural Network** | hidden_layer_sizes, activation, alpha |

All models are tuned with `RandomizedSearchCV` (5 iterations, 3-fold cross-validation, F1-macro scoring) and evaluated on a held-out test set.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lab-work-2.git
   cd lab-work-2
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn joblib
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Lab_work_2.ipynb
   ```

> ⚠️ Update `DATA_PATH` in cell 3 to the local path of `spotify_songs.csv`, or place it in the same directory as the notebook and use `DATA_PATH = 'spotify_songs.csv'`.

---

## Usage

Run all cells in `Lab_work_2.ipynb` top to bottom. The notebook is divided into clearly labelled sections:

| Section | Description |
|---------|-------------|
| **1. Imports** | All library imports |
| **2. Load Data** | Reads CSV, prints shape |
| **3. EDA** | Info, describe, null check |
| **4. Visualisations** | Genre dist., KDE, histograms, heatmap, Plotly charts |
| **5. Cleaning** | Drop nulls, encode target |
| **6. SMOTE** | Balance class distribution |
| **7. Scaling + PCA** | StandardScaler, PCA (95% variance) |
| **8. Model Setup** | Define models and hyperparameter grids |
| **9. Training & Tuning** | RandomizedSearchCV loop |
| **10. Evaluation** | Metrics per model, classification report |
| **11. Comparison** | Bar chart, radar plot, ROC curves, confusion matrices |
| **12. Save Best Model** | joblib export of best F1 model |

---

## Key Findings

- 🎯 **Ensemble methods** (Random Forest, Gradient Boosting) consistently outperform linear models for genre classification
- 📊 **EDM and rap** are the most distinguishable genres — high energy/danceability signatures
- 🔀 **SMOTE** meaningfully improved recall for minority genres (latin, r&b) without overfitting
- 🧩 **PCA** retained 95% of variance while significantly reducing dimensionality, speeding up training
- 📉 **Logistic Regression** had the highest LogLoss, confirming linear boundaries are insufficient for this task
- 🕸️ The **radar plot** showed that no single model dominates all metrics — trade-offs exist between accuracy, F1 and AUC depending on the use case
- 💾 The best model is automatically exported as `best_<ModelName>.pkl` for deployment

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Static plots |
| `seaborn` | Statistical visualisations |
| `plotly` | Interactive charts (parallel coords, sunburst, treemap) |
| `scikit-learn` | ML models, scaling, PCA, evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `joblib` | Model serialisation |

---

## References

- Spotify Songs dataset: [TidyTuesday 2020-01-21](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-01-21/readme.md)
- scikit-learn documentation: https://scikit-learn.org
- imbalanced-learn (SMOTE): https://imbalanced-learn.org
- Plotly Python docs: https://plotly.com/python

---

## License

This project is open-source and available under the [MIT License](LICENSE).
