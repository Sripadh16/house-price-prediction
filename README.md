# Advanced House Price Prediction & Real Estate Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark%20MLlib-E25A1C.svg)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end data science and Big Data analytics project developed for the **ASBD (Advanced Statistics & Big Data) / ADS (Applied Data Science)** course. This project addresses all remarks and rigorous academic criteria specified in course evaluations.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Review Feedback & Direct Resolution](#-review-feedback--direct-resolution)
3. [Dataset Architecture & Synthetic Pipeline](#-dataset-architecture--synthetic-pipeline)
4. [In-Depth Preprocessing Pipeline](#-in-depth-preprocessing-pipeline)
5. [Advanced Visualizations with Comparative Power](#-advanced-visualizations-with-comparative-power)
   - [Density with Rug Plot vs Traditional Histogram](#1-density-with-rug-plot-vs-traditional-histogram)
   - [Radar Chart (Spider Plot) Multivariate Comparison](#2-radar-chart-spider-plot-multivariate-comparison)
   - [Violin Plots & Correlation Matrices](#3-violin-plots--correlation-matrices)
6. [Depth of Predictive Analytics](#-depth-of-predictive-analytics)
   - [Regression Leaderboard (7 Algorithms)](#regression-leaderboard)
   - [Multi-Class Classification Depth (5 Algorithms)](#classification-leaderboard)
   - [Clustering Strategies (4 Frameworks)](#clustering-strategies)
   - [Association Rule Mining (Apriori vs FP-Growth)](#association-rule-mining)
7. [PySpark Big Data Analytics Exploration](#-pyspark-big-data-analytics-exploration)
8. [Repository Structure](#-repository-structure)
9. [Setup & Execution Guide](#-setup--execution-guide)

---

## 🎯 Project Overview
This project formulates property valuation as both a **continuous estimation (Regression)** and a **market tier categorization (Classification)** problem, augmented by **unsupervised customer/property segmentation (Clustering)**, **transactional basket association discovery (Rule Mining)**, and **distributed distributed processing (PySpark MLlib)**.

---

## 🔍 Review Feedback & Direct Resolution

| Instructor Remarks | Specific Requirement | Implementation & Resolution in this Repository |
| :--- | :--- | :--- |
| **"In-depth treatment of preprocessing techniques required"** | Move beyond basic missing value filling. Address outliers, feature engineering, skewness, and collinearity. | Applied IQR & Z-score detection with **Winsorization capping** [1st, 99th percentile], domain feature engineering (`HouseAge`, `TotalSF`, `TotalBaths`, `PricePerSqFt`, `QualityConditionIndex`), Log1p skewness reduction, and **Variance Inflation Factor (VIF)** diagnosis. |
| **"What about other plots such as radar, density rug, etc for their comparative power"** | High-dimensional and high-resolution exploratory visualizations with written comparative inferences. | Implemented **Density with Rug Plot** (`kdeplot` + `rugplot`) revealing individual tail observations across exterior qualities, and a **Radar Chart (Spider Plot)** benchmarking 6 continuous dimensions across market price tiers with deep analytical inferences. |
| **"What about depth of classifiers"** | Expand beyond a single basic Random Forest classifier. | Benchmarked 5 distinct classifier paradigms: **Random Forest, Gradient Boosting, Support Vector Machine (SVC), Multinomial Logistic Regression, and K-Nearest Neighbors (KNN)** with weighted Precision, Recall, F1, and normalized Confusion Matrices. |
| **"Depth of clustering strategies that are supported as packages!"** | Move beyond standalone K-Means; benchmark multiple clustering strategies. | Implemented **K-Means (Elbow + Silhouette)**, **Agglomerative Hierarchical Clustering (with Ward linkage Dendrogram)**, **DBSCAN (density & noise isolation)**, and **Gaussian Mixture Models (GMM)** with comparative 2D PCA projections. |
| **"What about rule mining techniques"** | Structured association discovery with algorithmic performance comparisons. | Discretized properties into transactional itemsets, benchmarked **Apriori vs FP-Growth** runtime efficiency, and extracted high-lift association rules (Support, Confidence, Lift, Leverage, Conviction). |
| **"No exploration in PySpark of the minimal predictive analytics techniques"** | Demonstrate distributed Big Data analytics. | Developed full **PySpark MLlib** pipeline with `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`, distributed **PySpark Regression**, **PySpark Classification**, and **PySpark Clustering**. |

---

## 📊 Dataset Architecture: Real Kaggle Ames Housing Benchmark
This repository utilizes the official **Kaggle Ames Housing Benchmark dataset (`train.csv`)** from the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

- **Total Transactions**: 1,460 genuine home sales in Ames, Iowa.
- **Dimensionality**: 81 raw attributes covering physical dimensions, zoning, exterior finishes, basement conditions, and mechanical systems.
- **Target Variable**: `SalePrice` ($ USD continuous valuation).
- **Synthetic Code Removal**: All prior mock/synthetic data generators have been completely deprecated and purged in favor of the authentic Kaggle benchmark.

---

## 🛠️ In-Depth Preprocessing Pipeline

1. **Systematic Missing Value Imputation**:
   - `LotFrontage`: Imputed with median of the respective `Neighborhood` to preserve local geometric parcel distributions.
   - Categorical Amenities (`Alley`, `BsmtQual`, `BsmtCond`, `FireplaceQu`, `GarageType`, `GarageFinish`, `PoolQC`, `Fence`, `MiscFeature`): Filled with `'None'`, explicitly representing lack of that amenity per official Ames documentation.
   - `MasVnrArea`: Imputed with 0; `GarageYrBlt`: Imputed with `YearBuilt`; `Electrical`: Imputed with mode.
2. **Outlier Mitigation via Winsorization**:
   - Rather than dropping extreme luxury records, continuous features exceeding $1.5 \times \text{IQR}$ are capped at the **1st and 99th percentiles**.
3. **Domain-Specific Feature Engineering**:
   - $\text{HouseAge} = 2024 - \text{YearBuilt}$
   - $\text{RemodAge} = 2024 - \text{YearRemodAdd}$
   - $\text{IsRemodeled} = \mathbb{I}(\text{YearRemodAdd} > \text{YearBuilt})$
   - $\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$
   - $\text{TotalBaths} = \text{FullBath} + 0.5 \times \text{HalfBath}$
   - $\text{PricePerSqFt} = \text{SalePrice} / (\text{GrLivArea} + 10^{-5})$
   - $\text{QualityConditionIndex} = \text{OverallQual} \times \text{OverallCond}$
4. **Variance Inflation Factor (VIF)**:
   - Diagnosed multicollinearity among composite indices (e.g. `QualityConditionIndex` VIF = 37.1) and mitigated variance inflation using regularized shrinkage ($L_1$ Lasso, $L_2$ Ridge, and ElasticNet).

---

## 📈 Advanced Visualizations with Comparative Power

### 1. Density with Rug Plot vs Traditional Histogram
![Density with Rug Plot](figures/02_density_rug_plot.png)
- **Comparative Power**: Unlike histograms which aggregate data into arbitrary, bin-width-dependent blocks, the **Rug Plot** places an exact tick mark for every property transaction along the axis. This reveals sample density, multi-modal clusters, and isolated outlier concentrations in the upper tails across exterior quality ratings (`Ex`, `Gd`, `TA`, `Fa`).

### 2. Radar Chart (Spider Plot) Multivariate Comparison
![Radar Chart](figures/03_radar_chart.png)
- **Comparative Power**: Conventional bivariate plots only compare two features simultaneously. The **Radar Chart** maps 6 normalized dimensions (`LotArea`, `OverallQual`, `OverallCond`, `GrLivArea`, `TotalBaths`, `GarageCars`) onto radial axes, clearly differentiating the holistic property profiles of **Affordable**, **Mid-Range**, and **Luxury** tiers at a single glance.

### 3. Violin Plots & Correlation Matrices
![Violin Plots](figures/04_violin_plots.png)
![Correlation Matrix](figures/05_correlation_heatmap.png)
- Showcases distribution shape, probability density, and quartile medians across diverse architectural styles.

---

## 🤖 Depth of Predictive Analytics

### Regression Leaderboard (Real Kaggle `train.csv`)
| Model | $R^2$ Score | Adjusted $R^2$ | RMSE ($) | MAE ($) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.9172** | **0.8978** | **$25,208** | **$16,050** | **9.97%** |
| **Random Forest Regressor** | **0.9073** | **0.8857** | **$26,668** | **$16,561** | **10.32%** |
| **Linear Regression (OLS)** | 0.8691 | 0.8386 | $31,683 | $19,267 | 11.28% |
| **Lasso Regression ($L_1$)** | 0.8686 | 0.8380 | $31,745 | $19,210 | 11.21% |
| **Ridge Regression ($L_2$)** | 0.8656 | 0.8343 | $32,109 | $19,317 | 11.28% |
| **Decision Tree Regressor** | 0.8625 | 0.8305 | $32,472 | $21,784 | 13.83% |
| **ElasticNet** | 0.8590 | 0.8261 | $32,891 | $19,433 | 11.27% |

![Regression Diagnostics](figures/06_regression_comparisons.png)
![Feature Importance](figures/07_feature_importance.png)

---

### Classification Leaderboard (Real Kaggle `train.csv`)
Formulation: Predicting property price category (**Affordable**, **Mid-Range**, **Luxury**).

| Classifier | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Classifier (SVC)** | **81.16%** | **81.58%** | **81.16%** | **0.8125** |
| **Gradient Boosting Classifier** | 79.79% | 80.11% | 79.79% | 0.7990 |
| **Random Forest Classifier** | 79.45% | 80.41% | 79.45% | 0.7971 |
| **Logistic Regression (Multinomial)**| 78.08% | 79.45% | 78.08% | 0.7847 |
| **K-Nearest Neighbors (KNN)** | 77.74% | 77.38% | 77.74% | 0.7749 |

![Classification Metrics](figures/08_classification_metrics_and_cm.png)

---

### Clustering Strategies
| Algorithm | Metric / Basis | Optimal Clusters / Output | Evaluation |
| :--- | :--- | :---: | :--- |
| **K-Means** | Euclidean Distance Partitioning | $k = 3$ | Silhouette Score = 0.279 (Elbow confirmed at $k=3$) |
| **Agglomerative Hierarchical** | Ward Linkage Minimum Variance | $k = 3$ | Dendrogram distance threshold confirmed |
| **Gaussian Mixture Models (GMM)**| Probabilistic Covariance Mixture | $k = 3$ | Silhouette Score = 0.178 |
| **DBSCAN** | Density Core Points ($\epsilon=1.2, \text{min}=8$)| 7 dense clusters | Identified 93 isolated noise properties |

![Dendrogram](figures/09_clustering_dendrogram.png)
![Clustering PCA Comparison](figures/10_clustering_pca_comparison.png)

---

### Association Rule Mining
- **Algorithmic Benchmark**: **Apriori vs FP-Growth** runtime efficiency benchmark on discretized transaction baskets.
- **Top Rules**:
  - $\text{IF } [\text{Age: Modern, Quality: High}] \implies \text{Price: Luxury}$ (Confidence: **96.0%**, Lift: **2.90**)
  - $\text{IF } [\text{Spacious Living Area, Luxury Baths}] \implies \text{Price: Luxury}$ (Confidence: **94.0%**, Lift: **2.84**)

---

## ⚡ PySpark Big Data Analytics Exploration
Implemented in `pyspark_house_prediction.py` and embedded in `HousePricePredictor.ipynb`:
- **Spark Architecture**: Leverages PySpark RDDs and DataFrames with Catalyst optimizer for out-of-core, distributed cluster computation.
- **MLlib Feature Pipeline**: Chained `StringIndexer` $\to$ `OneHotEncoder` $\to$ `VectorAssembler` $\to$ `StandardScaler`.
- **Distributed Evaluation**:
  - `LinearRegression`: $R^2 = 0.9124$, $\text{RMSE} = \$31,450.80$
  - `RandomForestRegressor`: $R^2 = 0.9248$, $\text{RMSE} = \$28,910.40$
  - `RandomForestClassifier`: $\text{Accuracy} = 0.8842$, $\text{F1} = 0.8821$
  - `KMeans`: $\text{Silhouette Evaluator} = 0.4285$

---

## 📁 Repository Structure
```text
.
├── train.csv                      # Official Kaggle Ames Housing dataset (1,460 rows, 81 cols)
├── midsem_version.ipynb           # Executed Midsem Notebook (streamlined, cut till classification)
├── HousePricePredictor.ipynb      # Complete Coursework Notebook (All 7 sections executed)
├── Midsem_Project_Report.docx     # Formal Midsem Evaluation Report with embedded figures
├── MIDSEM_DOCUMENT.md             # Academic evaluation documentation
├── run_full_analysis.py           # Standalone automated analysis pipeline script
├── pyspark_house_prediction.py    # PySpark Big Data MLlib pipeline
├── summary_metrics.json           # Exported JSON metric leaderboard
├── review.jpeg                    # Instructor feedback reference image
├── .gitignore                     # Git ignore rules
├── figures/                       # High-resolution generated visual assets (10 figures)
│   ├── 01_hist_box_kde.png
│   ├── 02_density_rug_plot.png
│   ├── 03_radar_chart.png
│   ├── 04_violin_plots.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_regression_comparisons.png
│   ├── 07_feature_importance.png
│   ├── 08_classification_metrics_and_cm.png
│   ├── 09_clustering_dendrogram.png
│   └── 10_clustering_pca_comparison.png
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Initialize Environment
```bash
git clone https://github.com/Sripadh16/house-price-prediction.git
cd house-price-prediction
pip install numpy pandas scipy scikit-learn matplotlib seaborn mlxtend statsmodels nbformat nbclient
```

### 2. Launch Midsem Submission Notebook
```bash
jupyter notebook midsem_version.ipynb
```

### 3. Run the Full Analytics Suite
```bash
python run_full_analysis.py
```

### 4. Run PySpark Predictive Analytics
```bash
python pyspark_house_prediction.py
```

