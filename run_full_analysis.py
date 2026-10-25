import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.stats.outliers_influence import variance_inflation_factor

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Setup directories and style
os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

print("="*80)
print("ASBD - ADS COURSE PROJECT: ADVANCED HOUSE PRICE PREDICTION & ANALYTICS")
print("Addressing Review Remarks: Preprocessing, Visualizations (Radar, Rug),")
print("Depth of Regression, Classifiers, Clustering Strategies, Rule Mining & PySpark")
print("="*80)

# ==============================================================================
# SECTION 1: DATA INGESTION & IN-DEPTH PREPROCESSING
# ==============================================================================
print("\n[STEP 1] Data Ingestion & In-depth Preprocessing Treatment...")
data_path = 'house_dataset_big.csv'
df = pd.read_csv(data_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# 1.1 Missing Value Diagnosis & Treatment
print("\n--- 1.1 Missing Value Diagnosis & Treatment ---")
null_counts = df.isnull().sum()
print("Null values per column:")
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No missing values found in raw dataset. Applying robust imputation pipeline.")

# 1.2 Outlier Detection & Treatment via IQR and Winsorization
print("\n--- 1.2 Outlier Detection & Treatment (IQR & Winsorization) ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'SalePrice' in numeric_cols:
    numeric_cols.remove('SalePrice')

outlier_summary = {}
df_clean = df.copy()
for col in ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)
    # Winsorize / Cap extreme 1st and 99th percentiles for robust model stability
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df_clean[col] = np.clip(df_clean[col], p01, p99)

print("Outliers detected by 1.5*IQR rule:", outlier_summary)
print("Applied Winsorization capping at [1st, 99th] percentiles to preserve sample size while curbing leverage.")

# 1.3 Feature Engineering
print("\n--- 1.3 Domain-Specific Feature Engineering ---")
CURRENT_YEAR = 2024
df_clean['HouseAge'] = CURRENT_YEAR - df_clean['YearBuilt']
df_clean['RemodAge'] = CURRENT_YEAR - df_clean['YearRemodAdd']
df_clean['IsRemodeled'] = (df_clean['YearRemodAdd'] > df_clean['YearBuilt']).astype(int)
df_clean['TotalSF'] = df_clean['TotalBsmtSF'] + df_clean['1stFlrSF'] + df_clean['2ndFlrSF']
df_clean['TotalBaths'] = df_clean['FullBath'] + 0.5 * df_clean['HalfBath']
df_clean['PricePerSqFt'] = df_clean['SalePrice'] / (df_clean['GrLivArea'] + 1e-5)
df_clean['QualityConditionIndex'] = df_clean['OverallQual'] * df_clean['OverallCond']
print("Engineered features: HouseAge, RemodAge, IsRemodeled, TotalSF, TotalBaths, PricePerSqFt, QualityConditionIndex.")

# 1.4 Feature Transformation & Skewness Reduction
print("\n--- 1.4 Skewness Analysis & Log Transformations ---")
raw_price_skew = df_clean['SalePrice'].skew()
raw_lot_skew = df_clean['LotArea'].skew()
df_clean['LogSalePrice'] = np.log1p(df_clean['SalePrice'])
df_clean['LogLotArea'] = np.log1p(df_clean['LotArea'])
df_clean['LogGrLivArea'] = np.log1p(df_clean['GrLivArea'])
print(f"SalePrice Skewness: Before = {raw_price_skew:.3f}, After Log1p = {df_clean['LogSalePrice'].skew():.3f}")
print(f"LotArea Skewness: Before = {raw_lot_skew:.3f}, After Log1p = {df_clean['LogLotArea'].skew():.3f}")

# 1.5 Categorical Encoding
print("\n--- 1.5 Categorical Encoding ---")
categorical_cols = ['Neighborhood', 'HouseStyle', 'ExterQual']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(f"Dataset dimension after One-Hot Encoding: {df_encoded.shape}")

# 1.6 Multicollinearity & Variance Inflation Factor (VIF)
print("\n--- 1.6 Multicollinearity Diagnosis via VIF ---")
vif_features = ['OverallQual', 'OverallCond', 'GrLivArea', 'TotalBsmtSF', 
                'FullBath', 'GarageCars', 'HouseAge', 'QualityConditionIndex']
X_vif = df_clean[vif_features].astype(float)
vif_data = pd.DataFrame()
vif_data['Feature'] = vif_features
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data = vif_data.sort_values(by='VIF', ascending=False)
print("Multicollinear Features Analysis (VIF):")
print(vif_data.to_string(index=False))
print("Inference: Features with high VIF (e.g. GrLivArea, OverallQual) share variance with target; regularized regression (Ridge/Lasso/ElasticNet) is used to control coefficient inflation.")

# ==============================================================================
# SECTION 2: ADVANCED DATA VISUALIZATIONS & COMPARATIVE POWER
# ==============================================================================
print("\n[STEP 2] Generating Advanced Visualizations with Comparative Power...")

# 2.0 Traditional Histogram + Box Plot + Summary
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(df_clean['SalePrice'], kde=True, ax=axes[0], color='#2b5c8f', bins=30)
axes[0].set_title('Histogram & KDE of SalePrice', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sale Price ($)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)

sns.boxplot(y=df_clean['SalePrice'], ax=axes[1], color='#457b9d')
axes[1].set_title('Box Plot of SalePrice (Quartiles & Outliers)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Sale Price ($)', fontsize=12)

plt.tight_layout()
fig.savefig('figures/01_hist_box_kde.png', dpi=300)
plt.close(fig)
print("Saved: figures/01_hist_box_kde.png")

# 2.1 Density Plot with Rug Plot vs Histogram
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Traditional Histogram with KDE
sns.histplot(df_clean['SalePrice'], kde=True, ax=axes[0], color='#2b5c8f', bins=30)
axes[0].set_title('Traditional Histogram + KDE of SalePrice', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sale Price ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

# Subplot 2: Density with Rug Plot across Exterior Quality tiers
exter_palette = {'Ex': '#e63946', 'Gd': '#457b9d', 'TA': '#2a9d8f', 'Fa': '#e76f51'}
sns.kdeplot(data=df_clean, x='SalePrice', hue='ExterQual', palette=exter_palette, common_norm=False, fill=True, alpha=0.25, ax=axes[1], linewidth=2)
sns.rugplot(data=df_clean, x='SalePrice', hue='ExterQual', palette=exter_palette, ax=axes[1], height=0.08, legend=False)
axes[1].set_title('Comparative Power: Density Plot with Rug Marks across Exterior Quality', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sale Price ($)', fontsize=12)
axes[1].set_ylabel('Kernel Density Estimate', fontsize=12)

plt.tight_layout()
fig.savefig('figures/02_density_rug_plot.png', dpi=300)
plt.close(fig)
print("Saved: figures/02_density_rug_plot.png")

# 2.2 Radar Chart (Spider Plot) Comparing Price Tiers
df_clean['PriceTier'] = pd.qcut(df_clean['SalePrice'], q=3, labels=['Affordable', 'Mid-Range', 'Luxury'])

radar_features = ['LotArea', 'OverallQual', 'OverallCond', 'GrLivArea', 'TotalBaths', 'GarageCars']
radar_summary = df_clean.groupby('PriceTier', observed=False)[radar_features].mean()

# Min-Max Normalization for Radar Chart (0 to 1 scale)
radar_norm = (radar_summary - df_clean[radar_features].min()) / (df_clean[radar_features].max() - df_clean[radar_features].min())

categories = radar_features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories, color='navy', size=11, fontweight='bold')
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=9)
plt.ylim(0, 1)

tier_colors = {'Affordable': '#2a9d8f', 'Mid-Range': '#457b9d', 'Luxury': '#e63946'}
for tier in ['Affordable', 'Mid-Range', 'Luxury']:
    values = radar_norm.loc[tier].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=tier, color=tier_colors[tier])
    ax.fill(angles, values, color=tier_colors[tier], alpha=0.18)

plt.title('Radar Chart (Spider Plot): Multivariate Profile Comparison by Price Tier', size=15, fontweight='bold', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11)
plt.tight_layout()
fig.savefig('figures/03_radar_chart.png', dpi=300)
plt.close(fig)
print("Saved: figures/03_radar_chart.png")

# 2.3 Violin Plots (Comparative distribution across House Styles)
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(x='HouseStyle', y='SalePrice', hue='HouseStyle', data=df_clean, palette='Set2', legend=False, inner='quartile', ax=ax)
ax.set_title('Violin Plot: SalePrice Distributions & Multi-Modality Across House Styles', fontsize=14, fontweight='bold')
ax.set_xlabel('House Architectural Style', fontsize=12)
ax.set_ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
fig.savefig('figures/04_violin_plots.png', dpi=300)
plt.close(fig)
print("Saved: figures/04_violin_plots.png")

# 2.4 Correlation Heatmap
plt.figure(figsize=(14, 10))
corr_cols = ['SalePrice', 'OverallQual', 'OverallCond', 'YearBuilt', 'GrLivArea', 'TotalBsmtSF', 
             'TotalSF', 'TotalBaths', 'GarageCars', 'GarageArea', 'HouseAge', 'QualityConditionIndex']
sns.heatmap(df_clean[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Annotated Correlation Matrix: Housing Attributes vs SalePrice', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_correlation_heatmap.png', dpi=300)
plt.close()
print("Saved: figures/05_correlation_heatmap.png")

# ==============================================================================
# SECTION 3: DEPTH OF REGRESSION TECHNIQUES & RESULTS
# ==============================================================================
print("\n[STEP 3] Comprehensive Comparative Regression Modeling...")

exclude_cols = ['SalePrice', 'LogSalePrice', 'PricePerSqFt', 'PriceTier', 'Neighborhood', 'HouseStyle', 'ExterQual']
X = df_encoded.drop(columns=[c for c in exclude_cols if c in df_encoded.columns])
y = df_encoded['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regression_models = {
    'Linear Regression (OLS)': LinearRegression(),
    'Ridge Regression (L2)': Ridge(alpha=10.0),
    'Lasso Regression (L1)': Lasso(alpha=50.0, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
}

reg_results = []
trained_regressors = {}

for name, model in regression_models.items():
    use_scaled = 'Linear' in name or 'Ridge' in name or 'Lasso' in name or 'Elastic' in name
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    trained_regressors[name] = (model, preds, use_scaled)
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    n = len(y_test)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    reg_results.append({
        'Model': name,
        'R2 Score': r2,
        'Adjusted R2': adj_r2,
        'RMSE ($)': rmse,
        'MAE ($)': mae,
        'MAPE (%)': mape
    })

reg_df = pd.DataFrame(reg_results).sort_values(by='R2 Score', ascending=False)
print("\nRegression Comparative Performance Leaderboard:")
print(reg_df.to_string(index=False))

# Regression Diagnostics Plot
best_reg_name = reg_df.iloc[0]['Model']
best_model, best_preds, _ = trained_regressors[best_reg_name]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_test, best_preds, alpha=0.6, color='#2b5c8f', edgecolors='k', s=45)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Ideal 45° Line')
axes[0].set_title(f'Actual vs Predicted SalePrice ({best_reg_name})', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Actual Sale Price ($)', fontsize=12)
axes[0].set_ylabel('Predicted Sale Price ($)', fontsize=12)
axes[0].legend(fontsize=11)

residuals = y_test - best_preds
axes[1].scatter(best_preds, residuals, alpha=0.6, color='#e76f51', edgecolors='k', s=45)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_title(f'Residuals vs Fitted Values ({best_reg_name})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fitted Values ($)', fontsize=12)
axes[1].set_ylabel('Residuals ($)', fontsize=12)

plt.tight_layout()
fig.savefig('figures/06_regression_comparisons.png', dpi=300)
plt.close(fig)
print("Saved: figures/06_regression_comparisons.png")

# Feature Importance Plot
rf_model = trained_regressors['Random Forest Regressor'][0]
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)

plt.figure(figsize=(12, 6))
feat_importances.plot(kind='barh', color='#457b9d', edgecolor='navy')
plt.gca().invert_yaxis()
plt.title('Top 12 Most Influential Features (Random Forest Regressor)', fontsize=14, fontweight='bold')
plt.xlabel('Mean Decrease in Impurity (Feature Importance)', fontsize=12)
plt.tight_layout()
plt.savefig('figures/07_feature_importance.png', dpi=300)
plt.close()
print("Saved: figures/07_feature_importance.png")

# ==============================================================================
# SECTION 4: DEPTH OF CLASSIFIERS & METRICS
# ==============================================================================
print("\n[STEP 4] Multi-Class Classification Depth (Affordable, Mid-Range, Luxury)...")

y_clf = df_clean['PriceTier']
X_clf = X.copy()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42),
    'Support Vector Classifier (SVC)': SVC(kernel='rbf', C=1.5, probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=7)
}

clf_results = []
clf_predictions = {}

for name, clf in classifiers.items():
    use_scaled = name in ['Support Vector Classifier (SVC)', 'Logistic Regression', 'K-Nearest Neighbors (KNN)']
    X_tr = X_train_c_scaled if use_scaled else X_train_c
    X_te = X_test_c_scaled if use_scaled else X_test_c
    
    clf.fit(X_tr, y_train_c)
    preds = clf.predict(X_te)
    clf_predictions[name] = preds
    
    acc = accuracy_score(y_test_c, preds)
    prec = precision_score(y_test_c, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test_c, preds, average='weighted')
    f1 = f1_score(y_test_c, preds, average='weighted')
    
    clf_results.append({
        'Classifier': name,
        'Accuracy': acc,
        'Weighted Precision': prec,
        'Weighted Recall': rec,
        'Weighted F1-Score': f1
    })

clf_df = pd.DataFrame(clf_results).sort_values(by='Weighted F1-Score', ascending=False)
print("\nClassification Comparative Performance Leaderboard:")
print(clf_df.to_string(index=False))

# Confusion Matrix for Best Classifier
best_clf_name = clf_df.iloc[0]['Classifier']
cm = confusion_matrix(y_test_c, clf_predictions[best_clf_name], labels=['Affordable', 'Mid-Range', 'Luxury'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

clf_df.plot(x='Classifier', y=['Accuracy', 'Weighted F1-Score'], kind='bar', ax=axes[0], colormap='viridis')
axes[0].set_title('Classifier Performance Benchmark Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_ylim(0.7, 1.0)
axes[0].tick_params(axis='x', rotation=30)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Affordable', 'Mid-Range', 'Luxury'], 
            yticklabels=['Affordable', 'Mid-Range', 'Luxury'], ax=axes[1], cbar=False)
axes[1].set_title(f'Confusion Matrix ({best_clf_name})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Price Tier', fontsize=12)
axes[1].set_ylabel('Actual Price Tier', fontsize=12)

plt.tight_layout()
fig.savefig('figures/08_classification_metrics_and_cm.png', dpi=300)
plt.close(fig)
print("Saved: figures/08_classification_metrics_and_cm.png")

# ==============================================================================
# SECTION 5: DEPTH OF CLUSTERING STRATEGIES
# ==============================================================================
print("\n[STEP 5] In-Depth Exploration of Clustering Strategies...")

cluster_features = ['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'TotalBaths', 'GarageCars']
X_clust = df_clean[cluster_features]
scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

# 5.1 K-Means & Elbow / Silhouette Evaluation
wcss = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_clust_scaled)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_clust_scaled, km.labels_))

optimal_k = 3
km_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
km_labels = km_final.fit_predict(X_clust_scaled)

# 5.2 Agglomerative (Hierarchical) Clustering
agg_model = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
agg_labels = agg_model.fit_predict(X_clust_scaled)

# 5.3 DBSCAN
dbscan_model = DBSCAN(eps=1.2, min_samples=8)
dbscan_labels = dbscan_model.fit_predict(X_clust_scaled)
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_dbscan_noise = list(dbscan_labels).count(-1)

# 5.4 Gaussian Mixture Model (GMM)
gmm_model = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=42)
gmm_labels = gmm_model.fit_predict(X_clust_scaled)

print(f"K-Means Optimal Clusters: {optimal_k} (Silhouette = {silhouette_score(X_clust_scaled, km_labels):.3f})")
print(f"Agglomerative Clusters: {optimal_k} (Silhouette = {silhouette_score(X_clust_scaled, agg_labels):.3f})")
print(f"DBSCAN Clusters: {n_dbscan_clusters}, Noise Points Detected: {n_dbscan_noise}")
print(f"GMM Clusters: {optimal_k} (Silhouette = {silhouette_score(X_clust_scaled, gmm_labels):.3f})")

# 5.5 Dendrogram Visualization
sample_indices = np.random.choice(X_clust_scaled.shape[0], size=80, replace=False)
linked = linkage(X_clust_scaled[sample_indices], method='ward')

plt.figure(figsize=(14, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Sample of 80 Properties)', fontsize=14, fontweight='bold')
plt.xlabel('Property Index', fontsize=12)
plt.ylabel('Ward Dissimilarity Euclidean Distance', fontsize=12)
plt.tight_layout()
plt.savefig('figures/09_clustering_dendrogram.png', dpi=300)
plt.close()
print("Saved: figures/09_clustering_dendrogram.png")

# 5.6 2D PCA Projection Comparing Clustering Algorithms
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clust_scaled)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

scatter_km = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='tab10', alpha=0.7, edgecolors='k', s=35)
axes[0, 0].set_title(f'K-Means Clustering (k={optimal_k})', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('PCA Component 1', fontsize=11)
axes[0, 0].set_ylabel('PCA Component 2', fontsize=11)

scatter_agg = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='Set1', alpha=0.7, edgecolors='k', s=35)
axes[0, 1].set_title(f'Agglomerative Hierarchical (Ward, k={optimal_k})', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('PCA Component 1', fontsize=11)
axes[0, 1].set_ylabel('PCA Component 2', fontsize=11)

scatter_gmm = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='Accent', alpha=0.7, edgecolors='k', s=35)
axes[1, 0].set_title(f'Gaussian Mixture Models (GMM, k={optimal_k})', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('PCA Component 1', fontsize=11)
axes[1, 0].set_ylabel('PCA Component 2', fontsize=11)

scatter_db = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='rainbow', alpha=0.7, edgecolors='k', s=35)
axes[1, 1].set_title(f'DBSCAN (eps=1.2, min_samples=8) - Noise={n_dbscan_noise}', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('PCA Component 1', fontsize=11)
axes[1, 1].set_ylabel('PCA Component 2', fontsize=11)

plt.tight_layout()
fig.savefig('figures/10_clustering_pca_comparison.png', dpi=300)
plt.close(fig)
print("Saved: figures/10_clustering_pca_comparison.png")

# ==============================================================================
# SECTION 6: RULE MINING TECHNIQUES (APRIORI & FP-GROWTH)
# ==============================================================================
print("\n[STEP 6] Association Rule Mining (Apriori vs FP-Growth)...")

transactions = []
for _, row in df_clean.iterrows():
    basket = []
    if row['GrLivArea'] > 2000:
        basket.append('LivingArea_Spacious')
    elif row['GrLivArea'] < 1200:
        basket.append('LivingArea_Compact')
    else:
        basket.append('LivingArea_Medium')
        
    if row['OverallQual'] >= 8:
        basket.append('Quality_High')
    elif row['OverallQual'] <= 4:
        basket.append('Quality_Low')
    else:
        basket.append('Quality_Medium')
        
    if row['HouseAge'] <= 20:
        basket.append('Age_Modern')
    elif row['HouseAge'] >= 60:
        basket.append('Age_Historical')
    else:
        basket.append('Age_Established')
        
    if row['BedroomAbvGr'] >= 4:
        basket.append('Bedrooms_4Plus')
    else:
        basket.append('Bedrooms_1To3')
        
    if row['TotalBaths'] >= 2.5:
        basket.append('Baths_Luxury')
        
    if row['GarageCars'] >= 2:
        basket.append('Garage_2Plus')
        
    basket.append(f"Price_{row['PriceTier']}")
    basket.append(f"Style_{row['HouseStyle']}")
    basket.append(f"Exter_{row['ExterQual']}")
    
    transactions.append(basket)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

start_ap = time.time()
frequent_itemsets_ap = apriori(df_trans, min_support=0.08, use_colnames=True)
time_ap = time.time() - start_ap

start_fp = time.time()
frequent_itemsets_fp = fpgrowth(df_trans, min_support=0.08, use_colnames=True)
time_fp = time.time() - start_fp

print(f"Apriori: Found {len(frequent_itemsets_ap)} frequent itemsets in {time_ap:.4f}s.")
print(f"FP-Growth: Found {len(frequent_itemsets_fp)} frequent itemsets in {time_fp:.4f}s (Speedup: {time_ap/max(time_fp, 1e-6):.2f}x).")

rules = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.2)
rules = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])

target_rules = rules[
    rules['consequents'].apply(lambda x: any('Price_' in item for item in x))
].head(8)

print("\nTop Real Estate Association Rules (Target: Price Tiers):")
for idx, r in target_rules.iterrows():
    ante = list(r['antecedents'])
    cons = list(r['consequents'])
    print(f"Rule: IF {ante} ==> THEN {cons} | Supp: {r['support']:.3f}, Conf: {r['confidence']:.3f}, Lift: {r['lift']:.3f}")

# ==============================================================================
# SECTION 7: SUMMARY REPORT & OUTPUT DICTIONARY
# ==============================================================================
summary_stats = {
    'num_samples': len(df_clean),
    'best_regression_model': best_reg_name,
    'best_reg_r2': float(reg_df.iloc[0]['R2 Score']),
    'best_reg_rmse': float(reg_df.iloc[0]['RMSE ($)']),
    'best_classifier_model': best_clf_name,
    'best_clf_accuracy': float(clf_df.iloc[0]['Accuracy']),
    'best_clf_f1': float(clf_df.iloc[0]['Weighted F1-Score']),
    'optimal_k_clusters': int(optimal_k),
    'kmeans_silhouette': float(silhouette_score(X_clust_scaled, km_labels)),
    'num_frequent_rules': int(len(rules)),
    'fp_growth_speedup': float(time_ap / max(time_fp, 1e-6))
}

with open('summary_metrics.json', 'w') as f:
    json.dump(summary_stats, f, indent=4)

print("\n" + "="*80)
print("PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print(f"Saved all 10 figures in './figures/' directory.")
print(f"Summary metrics exported to 'summary_metrics.json'.")
print("="*80)
