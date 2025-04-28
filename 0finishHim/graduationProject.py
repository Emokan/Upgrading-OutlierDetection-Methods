import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Input
from keras.api.optimizers import Adam
import keras_tuner as kt
import warnings

# Dataset and StandartScaler
digits = load_digits()
X, y = digits.data, digits.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# OCSVM Parameters
ocsvm_per_class_outliers = {}
ocsvm_total_outliers = 0
ocsvm_labels_all = np.zeros_like(y)
y_ocsvm_eval = np.ones_like(y)
param_grid = {'nu': [0.01, 0.05, 0.1, 0.7], 'kernel': ['poly', 'rbf', 'sigmoid', 'linear']}

# OCSVM
def ocswm_score_func(estimator, X):
    return np.mean(estimator.predict(X) == 1)

ocsvm_grid = GridSearchCV(OneClassSVM(), param_grid, cv=3, scoring=ocswm_score_func)
ocsvm_grid.fit(X_scaled)
best_ocsvm = ocsvm_grid.best_estimator_

# One-Class SVM
for i in range(10):
    class_indices = np.where(y == i)[0]
    X_class = X_scaled[class_indices]
    ocsvm_outliers = best_ocsvm.fit_predict(X_class)
    ocsvm_labels = np.where(ocsvm_outliers == 1, 1, 0)
    ocsvm_labels_all[class_indices] = ocsvm_labels
    y_ocsvm_eval[class_indices[ocsvm_labels == 0]] = 0
    ocsvm_per_class_outliers[i] = np.sum(ocsvm_labels == 0)
    ocsvm_total_outliers += np.sum(ocsvm_labels == 0)

# LOF - Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20)
lof_outliers = lof.fit_predict(X_scaled)
lof_labels = np.where(lof_outliers == -1, 0, 1)

# Isolation Forest
iforest = IsolationForest(contamination=0.05)
iforest_outliers = iforest.fit_predict(X_scaled)
iforest_labels = np.where(iforest_outliers == -1, 0, 1)

# Elliptic Envelope
envelope = EllipticEnvelope(contamination=0.05)
warnings.filterwarnings("ignore", message="The covariance matrix associated to your dataset is not full rank")
envelope_outliers = envelope.fit_predict(X_scaled)
envelope_labels = np.where(envelope_outliers == -1, 0, 1)

# CNN
y_with_outliers = np.where(ocsvm_labels_all == 0, 10, y)

all_outliers = np.union1d(np.union1d(np.union1d(np.where(ocsvm_labels_all == 0)[0],
                                                np.where(lof_labels == 0)[0]),
                                     np.where(iforest_labels == 0)[0]),
                          np.where(envelope_labels == 0)[0])

y_with_outliers[all_outliers] = 10

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_with_outliers, test_size=0.4, random_state=42)

X_train_combined = np.hstack([X_train, lof_labels[:len(X_train)].reshape(-1, 1),
                              iforest_labels[:len(X_train)].reshape(-1, 1),
                              envelope_labels[:len(X_train)].reshape(-1, 1)])

X_test_combined = np.hstack([X_test, lof_labels[:len(X_test)].reshape(-1, 1),
                             iforest_labels[:len(X_test)].reshape(-1, 1),
                             envelope_labels[:len(X_test)].reshape(-1, 1)])

# CNN Model
class CNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(X_train_combined.shape[1],)))
        model.add(Dense(units=hp.Int('units_1', min_value=64, max_value=256, step=64), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dense(11, activation='softmax'))  # 10 classes (0-9) + 1 for outliers (10)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

tuner = kt.RandomSearch(
    CNNHyperModel(),
    objective=kt.Objective("val_accuracy", direction="max"),
    max_trials=16,
    executions_per_trial=3,
    directory='my_dir',
    project_name='digit_classifier',
    overwrite=False
)

tuner.search(X_train_combined, y_train, epochs=20, batch_size=32, validation_data=(X_test_combined, y_test), verbose=1)
best_cnn_model = tuner.get_best_models(num_models=1)[0]
best_cnn_model.optimizer = None
best_cnn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# CNN Test
test_loss, test_accuracy = best_cnn_model.evaluate(X_test_combined, y_test, verbose=1)
predictions = np.argmax(best_cnn_model.predict(X_test_combined), axis=1)
predictions_with_outliers = np.where(predictions == 10, 10, predictions)

cnn_precision = precision_score(y_test, predictions_with_outliers, average='weighted')
cnn_recall = recall_score(y_test, predictions_with_outliers, average='weighted')
cnn_f1 = f1_score(y_test, predictions_with_outliers, average='weighted')


cnn_outliers = np.where(predictions_with_outliers == 10)[0]
ocsvm_outliers = np.where(y_test == 10)[0]
lof_outliers = np.where(lof_labels == 0)[0]
iforest_outliers = np.where(iforest_labels == 0)[0]
envelope_outliers = np.where(envelope_labels == 0)[0]
cnn_only_outliers = np.setdiff1d(cnn_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers), envelope_outliers))
ocsvm_only_outliers = np.setdiff1d(ocsvm_outliers, np.union1d(np.union1d(np.union1d(cnn_outliers, lof_outliers), iforest_outliers), envelope_outliers))
lof_only_outliers = np.setdiff1d(lof_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, cnn_outliers), iforest_outliers), envelope_outliers))
eliptik_only_outliers = np.setdiff1d(envelope_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers), cnn_outliers))
iso_only_outliers = np.setdiff1d(iforest_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), cnn_outliers), envelope_outliers))

# Terminal
print(f"OCSVM Total Detected Outliers: {ocsvm_total_outliers}")
for i in range(10):
    print(f"OCSVM Detected Outliers in Class {i}: {ocsvm_per_class_outliers[i]}")
print(f"LOF Detected Outliers: {np.sum(lof_labels == 0)}")
print(f"Isolation Forest Detected Outliers: {np.sum(iforest_labels == 0)}")
print(f"Elliptic Envelope Detected Outliers: {np.sum(envelope_labels == 0)}")
print(f"CNN Detected Outliers: {len(cnn_outliers)}")
print(f"CNN Only Detected Outliers: {len(cnn_only_outliers)}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {cnn_precision:.4f}")
print(f"Recall: {cnn_recall:.4f}")
print(f"F1-Score: {cnn_f1:.4f}")
print({len(eliptik_only_outliers)})
print({len(iso_only_outliers)})
print({len(ocsvm_only_outliers)})
print({len(lof_only_outliers)})


# Graph
plt.style.use('ggplot')

def gradient_text(x, y, text, ax=None, spacing=0.1, **kwargs):
    ax = ax or plt.gca()
    fontsize = kwargs.pop("fontsize", 16)
    weight = kwargs.pop("weight", "bold")
    cmap = plt.get_cmap("viridis")

    for i, char in enumerate(text):
        ax.text(x + i * spacing, y, char, color=cmap(i / len(text)), fontsize=fontsize, weight=weight)

fig, ax = plt.subplots(figsize=(12, 10))

normal_indices = np.where(y_with_outliers != 10)[0]
outlier_indices = np.where(y_with_outliers == 10)[0]

jitter = np.random.normal(0, 0.02, size=X_pca.shape)

# Data
ax.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], c='black', s=20, label='Normal', alpha=0.7)
ax.scatter(X_pca[ocsvm_outliers, 0] + jitter[ocsvm_outliers, 0],
           X_pca[ocsvm_outliers, 1] + jitter[ocsvm_outliers, 1],
           c='red', s=30, label='OCSVM Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[lof_outliers, 0] + jitter[lof_outliers, 0],
           X_pca[lof_outliers, 1] + jitter[lof_outliers, 1],
           c='orange', s=30, label='LOF Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[iforest_outliers, 0] + jitter[iforest_outliers, 0],
           X_pca[iforest_outliers, 1] + jitter[iforest_outliers, 1],
           c='orange', s=30, label='Isolation Forest Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[envelope_outliers, 0] + jitter[envelope_outliers, 0],
           X_pca[envelope_outliers, 1] + jitter[envelope_outliers, 1],
           c='orange', s=30, label='Elliptic Envelope Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[cnn_only_outliers, 0] + jitter[cnn_only_outliers, 0],
           X_pca[cnn_only_outliers, 1] + jitter[cnn_only_outliers, 1],
           c='blue', s=50, label='CNN Extra Outliers', edgecolors='white', linewidth=0.5, alpha=0.9, zorder=4)

gradient_text(-5, 10.5, "Outlier Detection Results", ax=ax, spacing=0.3)

ax.set_xlabel("PCA Component 1", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("PCA Component 2", fontsize=14, fontweight='bold', labelpad=10)

ax.legend(fontsize=12, loc='upper left', frameon=True)

total_outliers = len(np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers),
                                np.union1d(envelope_outliers, cnn_only_outliers)))
text = (f"OCSVM Detected Outliers: {len(ocsvm_outliers)}\n"
        f"LOF Detected Outliers: {len(lof_outliers)}\n"
        f"Isolation Forest Detected Outliers: {len(iforest_outliers)}\n"
        f"Elliptic Envelope Detected Outliers: {len(envelope_outliers)}\n"
        f"CNN Detected Outliers: {len(cnn_outliers)}\n"
        f"CNN Only Detected Outliers: {len(cnn_only_outliers)}\n"
        f"Total Detected Outliers by All Methods: {total_outliers}\n"
        f"Model Accuracy: {test_accuracy:.4f}")
ax.text(0.5, -0.25, text, transform=ax.transAxes, ha='center', va='top', fontsize=12)

# Grid
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()