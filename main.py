import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from scipy import stats
# Function to calculate Precision, Recall, F1-Score
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset and reduce it to 2D using PCA
data = load_breast_cancer()
X = data.data
y = data.target

# Data Normalization or Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection - PCA (Dimensionality Reduction)
pca = PCA(n_components=10)  # Reduce to the first 10 components
X_pca = pca.fit_transform(X_scaled)

# Alternative Feature Selection - RFE (Recursive Feature Elimination)
estimator = LogisticRegression(max_iter=10000)
selector = RFE(estimator, n_features_to_select=5)  # Select the best 5 features
X_rfe = selector.fit_transform(X_scaled, y)

# Before starting anomaly detection, we will reduce the data to 2D using PCA
X_final = X_pca  # Alternatively, you can use X_rfe here

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05)
iso_forest_outliers = iso_forest.fit_predict(X_final)

# LOF (Local Outlier Factor)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_outliers = lof.fit_predict(X_final)

# OCSVM (One-Class SVM)
ocsvm = OneClassSVM(nu=0.05)
ocsvm_outliers = ocsvm.fit_predict(X_final)

# Anomaly Detection using IQR (Interquartile Range)
Q1 = np.percentile(X_final, 25, axis=0)
Q3 = np.percentile(X_final, 75, axis=0)
IQR = Q3 - Q1
iqr_outliers = ((X_final < (Q1 - 1.5 * IQR)) | (X_final > (Q3 + 1.5 * IQR))).any(axis=1)

# Anomaly Detection using Z-Score
z_scores = np.abs(stats.zscore(X_final))
zscore_outliers = (z_scores > 3).any(axis=1)

# Let's assume the true labels as anomalies (for comparison with labeled data)
# For example, consider "1" as anomaly and "0" as normal.
y_true = np.random.choice([0, 1], size=len(X_final))  # Random labels, substitute with actual labels if available

# Creating the plots
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Function to calculate Precision, Recall, F1-Score and print them
def print_metrics(model_name, y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)  # Added zero_division=1 to avoid division by zero
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)  # Added zero_division=1 to avoid division by zero
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)  # Added zero_division=1 to avoid division by zero
    print(f"{model_name} : Precision: {precision*100:.2f}% , Recall: {recall*100:.2f}% , F1-Score: {f1*100:.2f}%")

# Isolation Forest Plot
axs[0, 0].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[0, 0].scatter(X_final[iso_forest_outliers == -1, 0], X_final[iso_forest_outliers == -1, 1], color='r', label='Outliers', s=8)
axs[0, 0].set_title('Isolation Forest', fontsize=12)
axs[0, 0].legend(fontsize=8)
axs[0, 0].text(0.5, -0.2, f"Outliers: {np.sum(iso_forest_outliers == -1)}",
               transform=axs[0, 0].transAxes, ha='center', fontsize=8)
print_metrics("Isolation Forest", y_true, iso_forest_outliers)

# LOF Plot
axs[0, 1].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[0, 1].scatter(X_final[lof_outliers == -1, 0], X_final[lof_outliers == -1, 1], color='r', label='Outliers', s=8)
axs[0, 1].set_title('Local Outlier Factor (LOF)', fontsize=12)
axs[0, 1].legend(fontsize=8)
axs[0, 1].text(0.5, -0.2, f"Outliers: {np.sum(lof_outliers == -1)}",
               transform=axs[0, 1].transAxes, ha='center', fontsize=8)
print_metrics("Local Outlier Factor (LOF)", y_true, lof_outliers)

# OCSVM Plot
axs[1, 0].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[1, 0].scatter(X_final[ocsvm_outliers == -1, 0], X_final[ocsvm_outliers == -1, 1], color='r', label='Outliers', s=8)
axs[1, 0].set_title('One-Class SVM (OCSVM)', fontsize=12)
axs[1, 0].legend(fontsize=8)
axs[1, 0].text(0.5, -0.2, f"Outliers: {np.sum(ocsvm_outliers == -1)}",
               transform=axs[1, 0].transAxes, ha='center', fontsize=8)
print_metrics("One-Class SVM (OCSVM)", y_true, ocsvm_outliers)

# IQR Plot
axs[1, 1].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[1, 1].scatter(X_final[iqr_outliers, 0], X_final[iqr_outliers, 1], color='r', label='Outliers', s=8)
axs[1, 1].set_title('Interquartile Range (IQR)', fontsize=12)
axs[1, 1].legend(fontsize=8)
axs[1, 1].text(0.5, -0.2, f"Outliers: {np.sum(iqr_outliers)}",
               transform=axs[1, 1].transAxes, ha='center', fontsize=8)
print_metrics("Interquartile Range (IQR)", y_true, iqr_outliers)

# Z-score Plot
axs[2, 0].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[2, 0].scatter(X_final[zscore_outliers, 0], X_final[zscore_outliers, 1], color='r', label='Outliers', s=8)
axs[2, 0].set_title('Z-score', fontsize=12)
axs[2, 0].legend(fontsize=8)
axs[2, 0].text(0.5, -0.2, f"Outliers: {np.sum(zscore_outliers)}",
               transform=axs[2, 0].transAxes, ha='center', fontsize=8)
print_metrics("Z-score", y_true, zscore_outliers)

# Identify the common outliers detected by all methods (i.e., those detected by all 5 methods)
combined_outliers = np.logical_or.reduce([iso_forest_outliers == -1, lof_outliers == -1, ocsvm_outliers == -1, iqr_outliers, zscore_outliers])

# Common outliers (those detected by all 5 methods)
common_outliers = np.logical_and.reduce([iso_forest_outliers == -1, lof_outliers == -1, ocsvm_outliers == -1, iqr_outliers, zscore_outliers])

# In the last plot, we will show all and common outliers in green
axs[2, 1].scatter(X_final[:, 0], X_final[:, 1], color='b', label='Normal', s=8)
axs[2, 1].scatter(X_final[combined_outliers, 0], X_final[combined_outliers, 1], color='r', label='All', s=8)
axs[2, 1].scatter(X_final[common_outliers, 0], X_final[common_outliers, 1], color='g', label='Common Outliers', s=8)
axs[2, 1].set_title('Combined Outliers', fontsize=12)
axs[2, 1].legend(fontsize=8)
axs[2, 1].text(0.5, -0.2, f"Outliers: {np.sum(combined_outliers)}",
               transform=axs[2, 1].transAxes, ha='center', fontsize=8)
print(f"Common Outliers: {np.sum(common_outliers)}")

# Show all plots
plt.tight_layout()
plt.show()

print("Isolation Forest - Outliers:", np.sum(iso_forest_outliers == -1))
print("LOF - Outliers:", np.sum(lof_outliers == -1))
print("OCSVM - Outliers:", np.sum(ocsvm_outliers == -1))
print("IQR - Outliers:", np.sum(iqr_outliers))
print("Z-score - Outliers:", np.sum(zscore_outliers))
print("(LOF + Isolation Forest + OCSVM + IQR + Z-score) Outliers:", np.sum(combined_outliers))
print("(LOF + Isolation Forest + OCSVM + IQR + Z-score) Common Outliers:", np.sum(common_outliers))
