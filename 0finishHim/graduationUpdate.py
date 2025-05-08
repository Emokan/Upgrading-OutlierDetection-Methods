import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten, Concatenate
from keras.api.layers import RandomRotation, RandomZoom, RandomTranslation
from keras.api.optimizers import Adam
from keras.api.utils import to_categorical
import seaborn as sns
import warnings
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.layers import BatchNormalization

# Dataset and StandardScaler
digits = load_digits()
X, y = digits.data, digits.target       #moduled 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)    

# PCA
pca = PCA(n_components=2)                   #moduled 
X_pca = pca.fit_transform(X_scaled)

# OCSVM Model Güncellemesi
ocsvm_per_class_outliers = {}
ocsvm_total_outliers = 0
ocsvm_labels_all = np.zeros_like(y)
y_ocsvm_eval = np.ones_like(y)
param_grid = {'nu': [0.01, 0.05, 0.1, 0.7], 'kernel': ['poly', 'rbf', 'sigmoid', 'linear']}

# OCSVM - GridSearchCV ile modelin en iyi parametrelerini bulma
def ocswm_score_func(estimator, X):
    return np.mean(estimator.predict(X) == 1)

ocsvm_grid = GridSearchCV(OneClassSVM(), param_grid, cv=3, scoring=ocswm_score_func)
ocsvm_grid.fit(X_scaled)
best_ocsvm = ocsvm_grid.best_estimator_

# OCSVM - Anomali Tespiti ve Etiketleme
for i in range(10):
    class_indices = np.where(y == i)[0]
    X_class = X_scaled[class_indices]
    ocsvm_outliers = best_ocsvm.fit_predict(X_class)
    ocsvm_labels = np.where(ocsvm_outliers == 1, 1, 0)
    ocsvm_labels_all[class_indices] = ocsvm_labels
    y_ocsvm_eval[class_indices[ocsvm_labels == 0]] = 0
    ocsvm_per_class_outliers[i] = np.sum(ocsvm_labels == 0)
    ocsvm_total_outliers += np.sum(ocsvm_labels == 0)

# Skor Eşiği (Threshold) - OCSVM için
ocsvm_scores = best_ocsvm.decision_function(X_scaled)
threshold = np.percentile(ocsvm_scores, 5)  # %5'lik kesme noktasını kullan
ocsvm_labels_thresholded = np.where(ocsvm_scores < threshold, 0, 1)  # Skor eşik değerinin altındaki veriler anomali olarak etiketlenir
ocsvm_total_outliers_thresholded = np.sum(ocsvm_labels_thresholded == 0)

# OCSVM Test Sonuçları (Threshold ile)
print(f"OCSVM (Threshold) Total Detected Outliers: {ocsvm_total_outliers_thresholded}")
for i in range(10):
    print(f"OCSVM (Threshold) Detected Outliers in Class {i}: {np.sum(ocsvm_labels_thresholded[y == i] == 0)}")
    
# OCSVM Doğruluk Metriklerini Hesapla
ocsvm_accuracy = accuracy_score(y_ocsvm_eval, np.where(ocsvm_labels_thresholded == 0, 0, 1))
print(f"OCSVM Accuracy: {ocsvm_accuracy:.4f}")

# LOF - Local Outlier Factor - Tutorial'daki yaklaşıma göre güncellendi
lof = LocalOutlierFactor(n_neighbors=17, contamination=0.05)  # Tutorial'daki default değerlere yakın
lof_scores = -lof.fit_predict(X_scaled)  # Negatif değerleri alıyoruz, sklearn'de -1=anomali, 1=normal            +moduled
# LOF puanlarını tutorial formatına göre dönüştür (tutorial'da 1=anomali, 0=normal)
lof_labels = np.where(lof_scores > 0, 1, 0)  # Sklearn'de -1 olan anomalileri 1'e çeviriyoruz (tutorial formatı)

# Tutorial'da olduğu gibi anomali sayısını ve oranını raporla
print(f"LOF Detected Anomaly Count: {np.sum(lof_labels == 1)}")                                      #moduled
print(f"LOF Anomaly Percentage: {np.sum(lof_labels == 1) / len(lof_labels) * 100:.2f}%")

# LOF Doğruluk Metriklerini Hesapla
y_lof_eval = np.ones_like(y)
y_lof_eval[lof_labels == 1] = 0                               #moduled
lof_accuracy = accuracy_score(y_ocsvm_eval, y_lof_eval)
print(f"LOF Accuracy: {lof_accuracy:.4f}")

# Isolation Forest - Tutorial'daki yaklaşıma göre güncellendi
outlier_fraction = 0.05  # Dataset içindeki anomali oranını temsil eder

# Isolation Forest modeli - tutorial parametreleriyle
iforest = IsolationForest(n_estimators=100, max_samples=len(X_scaled), 
                         contamination=outlier_fraction, random_state=42, verbose=0)

# Model eğitimi ve tahmin - tutorial yaklaşımına uygun olarak ayrı adımlar
iforest.fit(X_scaled)  # Önce modeli eğit
iforest_scores = iforest.decision_function(X_scaled)  # Anomali skorlarını al
iforest_outliers = iforest.predict(X_scaled)  # Tahminleri al

# Tutorial'dakiyle aynı mantıkta değerleri dönüştürme
# Sklearn'de Isolation Forest çıktısı: 1 (normal), -1 (outlier)
# İstenen format: 1 (normal), 0 (outlier)
iforest_labels = np.where(iforest_outliers == -1, 0, 1)

# Isolation Forest Doğruluk Metriklerini Hesapla
y_iforest_eval = np.ones_like(y)
y_iforest_eval[iforest_labels == 0] = 0
iforest_accuracy = accuracy_score(y_ocsvm_eval, y_iforest_eval)
print(f"Isolation Forest Accuracy: {iforest_accuracy:.4f}")

# Elliptic Envelope - Tutorial'daki yaklaşıma göre güncellendi
# Tutorial'da olduğu gibi contamination değeri kullanarak model oluştur
envelope = EllipticEnvelope(contamination=0.1, random_state=2)  # Tutorial'daki parametreler

# Uyarıları bastır (tutorial'da bu uyarı yoktu)
warnings.filterwarnings("ignore", message="The covariance matrix associated to your dataset is not full rank")

# Tutorial'da olduğu gibi tek adımda fit ve predict yap
envelope_outliers = envelope.fit_predict(X_scaled)

# Tutorial'da olduğu gibi -1 (aykırı) ve 1 (normal) olarak etiketlenmiş durumda
# Kodun geri kalanı için 0 (aykırı) ve 1 (normal) formatına dönüştür
envelope_labels = np.where(envelope_outliers == -1, 0, 1)

# Tutorial'da olduğu gibi aykırı değer sayısını ve oranını hesapla
envelope_outlier_count = np.sum(envelope_labels == 0)
envelope_outlier_ratio = envelope_outlier_count / len(X_scaled)
print(f"Elliptic Envelope Detected Outliers: {envelope_outlier_count}")
print(f"Elliptic Envelope Outlier Ratio: {envelope_outlier_ratio:.2%}")

# Elliptic Envelope Doğruluk Metriklerini Hesapla
y_envelope_eval = np.ones_like(y)
y_envelope_eval[envelope_labels == 0] = 0
envelope_accuracy = accuracy_score(y_ocsvm_eval, y_envelope_eval)
print(f"Elliptic Envelope Accuracy: {envelope_accuracy:.4f}")

# Tüm anomali etiketlerini birleştir
y_with_outliers = np.copy(y)

# Tüm outlier'ları birleştir
all_outliers = np.union1d(np.union1d(np.union1d(np.where(ocsvm_labels_thresholded == 0)[0],
                                                np.where(lof_labels == 1)[0]),  # Tutorial formatına göre 1=anomali
                                     np.where(iforest_labels == 0)[0]),
                          np.where(envelope_labels == 0)[0])

# Outlier'ların sınıfını 10 yap (yeni sınıf)
y_with_outliers[all_outliers] = 10

# 2 ve 8 rakamları için öznitelik çıkarma fonksiyonu
def extract_digit_features(images, reshape=True):
    """
    Enhanced feature extraction with special focus on differentiating digit 1 from 8
    
    Args:
        images: Image dataset
        reshape: Whether to reshape the images
        
    Returns:
        Extracted features
    """
    features = []
    
    for img in images:
        if reshape:
            img = img.reshape(8, 8)
        img = img.squeeze()  # Single channel image
        
        # Original features from your code
        horizontal_lines = np.sum(img > 0.5, axis=1)  # Active pixels per row
        vertical_lines = np.sum(img > 0.5, axis=0)    # Active pixels per column
        center_region = np.mean(img[3:5, 3:5])
        top_region = np.mean(img[0:3, :])
        middle_region = np.mean(img[3:5, :])
        bottom_region = np.mean(img[5:8, :])
        
        top_left = np.mean(img[0:2, 0:2])
        top_right = np.mean(img[0:2, 6:8])
        bottom_left = np.mean(img[6:8, 0:2])
        bottom_right = np.mean(img[6:8, 6:8])
        
        upper_loop = np.mean(img[1:3, 2:6])
        lower_loop = np.mean(img[5:7, 2:6])
        
        left_edge = np.sum(img[:, 0:2] > 0.5)
        right_edge = np.sum(img[:, 6:8] > 0.5)
        center_line = np.sum(img[:, 3:5] > 0.5)
        
        # NEW FEATURES FOR DIGIT 1 vs 8 DISTINCTION
        
        # 1. Aspect ratio - digit 1 is tall and narrow
        height = np.sum(np.any(img > 0.5, axis=1))
        width = np.sum(np.any(img > 0.5, axis=0))
        aspect_ratio = height / (width + 1e-6)  # Avoid division by zero
        
        # 2. Vertical symmetry - digit 8 should be more symmetric than 1
        left_half = img[:, :4]
        right_half = img[:, 4:]
        right_half_flipped = np.fliplr(right_half)
        vertical_symmetry = np.mean(np.abs(left_half - right_half_flipped))
        
        # 3. Center column density - digit 1 has high density in center columns
        center_column_density = np.mean(img[:, 3:5])
        
        # 4. Digit 1 specific feature - check for single vertical line
        vertical_line_check = np.mean([np.sum(img[i, 3:5] > 0.5) > 0 for i in range(8)])
        
        # Combine all features (original + new)
        feature_vector = np.concatenate([
            horizontal_lines, vertical_lines,
            [center_region, top_region, middle_region, bottom_region],
            [top_left, top_right, bottom_left, bottom_right],
            [upper_loop, lower_loop, left_edge, right_edge, center_line],
            # New features for 1 vs 8 distinction
            [aspect_ratio, vertical_symmetry, center_column_density, vertical_line_check]
        ])
        
        features.append(feature_vector)
    
    return np.array(features)
# CNN - Tutorial'a göre güncellendi
# Orijinal görüntüleri kullan (8x8)
X_images = digits.images / 16.0  # Tutorial'a göre normalizasyon
X_images = X_images.reshape(-1, 8, 8, 1)  # CNN için reshape

# Özel öznitelikleri çıkar
X_custom_features = extract_digit_features(digits.images)

# One-hot encoding yap - normal sınıflar (0-9) + outlier sınıfı (10)
# Toplam 11 sınıf olacak
y_categorical = to_categorical(y_with_outliers, num_classes=11)

# Veri setini böl - Tutorial'a göre
X_train_img, X_test_img, y_train_cat, y_test_cat = train_test_split(X_images, y_categorical, 
                                                                  test_size=0.4, random_state=42)

# Öznitelik veri setini de aynı şekilde böl
_, X_test_features, _, _ = train_test_split(X_custom_features, y_categorical, 
                                           test_size=0.4, random_state=42)

# Eğitim setinde 2 ve 8 rakamlarına özel sınıf ağırlıkları
# Önce orijinal sınıf etiketlerini bul
y_train_original = np.argmax(y_train_cat, axis=1)
# 2. MODIFY the class weights dictionary to increase weight for digit 1:
class_weights = {i: 1.0 for i in range(11)}
class_weights[1] = 2.5  # Add high weight for digit 1
class_weights[2] = 2.0  # Keep your original weight for 2
class_weights[8] = 2.0  # Keep your original weight for 8
class_weights[10] = 1.0  # Keep weight for outlier class if needed

# Hibrit model oluştur (CNN + Öznitelik tabanlı)
# Görüntü girişi
img_input = Input(shape=(8, 8, 1))

# Data Augmentation Katmanları
x = RandomRotation(0.05)(img_input)
x = RandomZoom(0.1)(x)
x = RandomTranslation(0.1, 0.1)(x)
x = BatchNormalization()(x)

# CNN bölümü
x = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
cnn_features = Dense(128, activation='relu')(x)
cnn_features = Dropout(0.3)(cnn_features)

# Öznitelik girişi
feature_input = Input(shape=(X_custom_features.shape[1],))
feature_dense = Dense(32, activation='relu')(feature_input)
feature_dense1 = Dense(64, activation='relu')(feature_input)  # Add intermediate layer
feature_dense1 = BatchNormalization()(feature_dense1)  # Add normalization
feature_dense = Dense(32, activation='relu')(feature_dense1)
# İki öznitelik grubunu birleştir
combined = Concatenate()([cnn_features, feature_dense])
combined = Dropout(0.5)(combined)
output = Dense(11, activation='softmax')(combined)

# Extra layer for combining features 
combined = Concatenate()([cnn_features, feature_dense])
combined = Dense(64, activation='relu')(combined)  # Add extra processing layer
combined = BatchNormalization()(combined)
combined = Dropout(0.4)(combined)
output = Dense(11, activation='softmax')(combined)

# Modeli oluştur
hybrid_model = Model(inputs=[img_input, feature_input], outputs=output)

# Compile
optimizer = Adam(learning_rate=0.001)
hybrid_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Özel öznitelikler için eğitim ve test veri setleri
X_train_features, X_test_custom_features, _, _ = train_test_split(X_custom_features, y_categorical, 
                                                                test_size=0.4, random_state=42)

#TRAINING 
epochs = 20  
batch_size = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = hybrid_model.fit(
    [X_train_img, X_train_features], y_train_cat,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr],
    validation_data=([X_test_img, X_test_custom_features], y_test_cat),
    class_weight=class_weights,
    verbose=1
)

# Hibrit modeli değerlendir
test_loss, test_accuracy = hybrid_model.evaluate([X_test_img, X_test_custom_features], y_test_cat, verbose=1)



# Modelin tahminlerini al
y_pred_proba = hybrid_model.predict([X_test_img, X_test_custom_features])
y_pred_classes = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# 5. ADD this function to analyze per-digit accuracy after evaluation:
def analyze_per_digit_accuracy(y_true, y_pred):
    """Analyze accuracy for each digit class"""
    print("\n===== Per-Digit Accuracy Analysis =====")
    for digit in range(10):  # For each digit
        idx = np.where(y_true == digit)[0]
        if len(idx) > 0:
            accuracy = np.mean(y_pred[idx] == y_true[idx])
            print(f"Accuracy for digit {digit}: {accuracy:.4f}")
    

# Call this function after getting predictions:
analyze_per_digit_accuracy(y_true, y_pred_classes)


# Tüm sınıflar için metrikler
cnn_precision = precision_score(y_true, y_pred_classes, average='weighted')
cnn_recall = recall_score(y_true, y_pred_classes, average='weighted')
cnn_f1 = f1_score(y_true, y_pred_classes, average='weighted')

# Ayrıştırılmış doğruluk metrikleri
normal_indices = np.where(y_true != 10)[0]
normal_accuracy = accuracy_score(y_true[normal_indices], y_pred_classes[normal_indices])

# Aykırı değerler için doğruluk (sadece sınıf 10)
outlier_indices = np.where(y_true == 10)[0]
if len(outlier_indices) > 0:
    outlier_accuracy = accuracy_score(y_true[outlier_indices], y_pred_classes[outlier_indices])
else:
    outlier_accuracy = 0.0

# Aykırı değer tespiti için precision ve recall
outlier_precision = precision_score(y_true == 10, y_pred_classes == 10)
outlier_recall = recall_score(y_true == 10, y_pred_classes == 10)

# Confusion Matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# CNN tarafından tespit edilen outlier'ları bul
cnn_outliers = np.where(y_pred_classes == 10)[0]
true_outliers = np.where(y_true == 10)[0]
cnn_only_outliers = np.setdiff1d(cnn_outliers, true_outliers)

# Diğer metodlar tarafından tespit edilen outlier'lar
ocsvm_outliers = np.where(y_with_outliers == 10)[0]
lof_outliers = np.where(lof_labels == 1)[0]
iforest_outliers = np.where(iforest_labels == 0)[0]
envelope_outliers = np.where(envelope_labels == 0)[0]

# Sadece belirli bir metod tarafından tespit edilenler
cnn_only_outliers = np.setdiff1d(cnn_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers), envelope_outliers))
ocsvm_only_outliers = np.setdiff1d(ocsvm_outliers, np.union1d(np.union1d(np.union1d(cnn_outliers, lof_outliers), iforest_outliers), envelope_outliers))
lof_only_outliers = np.setdiff1d(lof_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, cnn_outliers), iforest_outliers), envelope_outliers))
eliptik_only_outliers = np.setdiff1d(envelope_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), iforest_outliers), cnn_outliers))
iso_only_outliers = np.setdiff1d(iforest_outliers, np.union1d(np.union1d(np.union1d(ocsvm_outliers, lof_outliers), cnn_outliers), envelope_outliers))

# Terminal sonuçları yazdır
print(f"OCSVM Total Detected Outliers: {ocsvm_total_outliers}")
for i in range(10):
    print(f"OCSVM Detected Outliers in Class {i}: {ocsvm_per_class_outliers[i]}")
print(f"LOF Detected Outliers: {np.sum(lof_labels == 1)}")
print(f"Isolation Forest Detected Outliers: {np.sum(iforest_labels == 0)}")
print(f"Elliptic Envelope Detected Outliers: {np.sum(envelope_labels == 0)}")
print(f"CNN Detected Outliers: {len(cnn_outliers)}")
print(f"CNN Only Detected Outliers: {len(cnn_only_outliers)}")
print(f"CNN Test Accuracy: {test_accuracy:.4f}")
print(f"CNN Precision: {cnn_precision:.4f}")
print(f"CNN Recall: {cnn_recall:.4f}")
print(f"CNN F1-Score: {cnn_f1:.4f}")
print(f"CNN Normal Digits Accuracy: {normal_accuracy:.4f}")
print(f"CNN Outlier Detection Accuracy: {outlier_accuracy:.4f}")
print(f"CNN Outlier Precision: {outlier_precision:.4f}")
print(f"CNN Outlier Recall: {outlier_recall:.4f}")
print(f"Elliptic Envelope Only Outliers: {len(eliptik_only_outliers)}")
print(f"Isolation Forest Only Outliers: {len(iso_only_outliers)}")
print(f"OCSVM Only Outliers: {len(ocsvm_only_outliers)}")
print(f"LOF Only Outliers: {len(lof_only_outliers)}")

# Loss grafiği
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.plot(history.history['val_loss'], color='b', label="Validation Loss")
plt.plot(history.history['loss'], color='r', label="Training Loss")
plt.title("Hybrid Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy grafiği
plt.subplot(1, 3, 2)
plt.plot(history.history['val_accuracy'], color='b', label="Validation Accuracy")
plt.plot(history.history['accuracy'], color='r', label="Training Accuracy")
plt.title("Hybrid Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Confusion Matrix grafiği
plt.subplot(1, 3, 3)
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()

# Özel bir fonksiyon: 2 ve 8 için hatalı sınıflandırmaları görselleştirme
def visualize_misclassified_digits(X_test, y_true, y_pred, digit_class):
    # Belirli bir rakam sınıfını al
    class_indices = np.where(y_true == digit_class)[0]
    # Hatalı sınıflandırılanları bul
    misclassified = class_indices[y_pred[class_indices] != digit_class]
    
    if len(misclassified) == 0:
        print(f"No misclassified samples for digit {digit_class}")
        return
    
    # En fazla 10 tane göster
    n_samples = min(10, len(misclassified))
    plt.figure(figsize=(15, 6))
    
    for i in range(n_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[misclassified[i]].reshape(8, 8), cmap='gray')
        plt.title(f"True: {digit_class}, Pred: {y_pred[misclassified[i]]}")
        plt.axis('off')
    
    plt.suptitle(f"Misclassified Digit {digit_class} Samples")
    plt.tight_layout()
    plt.show()

# 2 ve 8 rakamları için hatalı sınıflandırmaları görselleştir
for i in range(10):
    visualize_misclassified_digits(X_test_img, y_true, y_pred_classes, i)

# Graph - Outlier Detection sonuçları
plt.figure(figsize=(12, 10))
ax = plt.gca()

normal_indices = np.where(y_with_outliers != 10)[0]
outlier_indices = np.where(y_with_outliers == 10)[0]

jitter = np.random.normal(0, 0.02, size=X_pca.shape)

# Data Plotting
ax.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], c='black', s=20, label='Normal', alpha=0.7)
ax.scatter(X_pca[ocsvm_labels_thresholded == 0, 0] + jitter[ocsvm_labels_thresholded == 0, 0],
           X_pca[ocsvm_labels_thresholded == 0, 1] + jitter[ocsvm_labels_thresholded == 0, 1],
           c='red', s=30, label='OCSVM Outliers (Threshold)', alpha=0.8, zorder=3)
ax.scatter(X_pca[lof_labels == 1, 0] + jitter[lof_labels == 1, 0],
           X_pca[lof_labels == 1, 1] + jitter[lof_labels == 1, 1],
           c='orange', s=30, label='LOF Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[iforest_labels == 0, 0] + jitter[iforest_labels == 0, 0],
           X_pca[iforest_labels == 0, 1] + jitter[iforest_labels == 0, 1],
           c='orange', s=30, label='Isolation Forest Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[envelope_labels == 0, 0] + jitter[envelope_labels == 0, 0],
           X_pca[envelope_labels == 0, 1] + jitter[envelope_labels == 0, 1],
           c='green', s=30, label='Elliptic Envelope Outliers', alpha=0.8, zorder=3)
ax.scatter(X_pca[cnn_only_outliers, 0] + jitter[cnn_only_outliers, 0],
           X_pca[cnn_only_outliers, 1] + jitter[cnn_only_outliers, 1],
           c='blue', s=50, label='CNN Extra Outliers', edgecolors='white', linewidth=0.5, alpha=0.9, zorder=4)

# Gradient metodu için
def gradient_text(x, y, text, ax=None, spacing=0.1, **kwargs):
    ax = ax or plt.gca()
    fontsize = kwargs.pop("fontsize", 16)
    weight = kwargs.pop("weight", "bold")
    cmap = plt.get_cmap("viridis")

    for i, char in enumerate(text):
        ax.text(x + i * spacing, y, char, color=cmap(i / len(text)), fontsize=fontsize, weight=weight)

gradient_text(-5, 10.5, "Hybrid Model Outlier Detection Results", ax=ax, spacing=0.3)

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
        f"CNN Accuracy: {test_accuracy:.4f}")
ax.text(0.5, -0.25, text, transform=ax.transAxes, ha='center', va='top', fontsize=12)

# Grid
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()