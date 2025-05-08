# main.py
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.api.utils import to_categorical

# Özel modülleri import et
from models import (
    create_ocsvm_model,
    create_lof_model,
    create_isolation_forest,
    create_elliptic_envelope,
    create_hybrid_model,
    train_hybrid_model
)
from utils import (
    extract_digit_features,
    analyze_per_digit_accuracy,
    visualize_misclassified_digits,
    plot_training_history,
    plot_confusion_matrix,
    plot_outlier_detection_results
)


def main():
    # Uyarıları bastır
    warnings.filterwarnings("ignore", message="The covariance matrix associated to your dataset is not full rank")
    
    print("=== MNIST Rakam Veri Seti Aykırı Değer Tespiti ve Sınıflandırma ===")
    
    # Veri setini yükle
    print("Veri seti yükleniyor...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Verileri normalleştir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA ile boyut indirgeme
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Anomali tespit modellerini oluştur
    print("Anomali tespit modelleri hazırlanıyor...")
    
    # 1. OCSVM Modeli
    ocsvm_grid = create_ocsvm_model()
    ocsvm_grid.fit(X_scaled)
    best_ocsvm = ocsvm_grid.best_estimator_
    print(f"En iyi OCSVM parametreleri: {ocsvm_grid.best_params_}")
    
    # OCSVM her sınıf için ayrı aykırı değer tespiti yap
    ocsvm_per_class_outliers = {}
    ocsvm_total_outliers = 0
    ocsvm_labels_all = np.zeros_like(y)
    y_ocsvm_eval = np.ones_like(y)
    
    for i in range(10):
        class_indices = np.where(y == i)[0]
        X_class = X_scaled[class_indices]
        ocsvm_outliers = best_ocsvm.fit_predict(X_class)
        ocsvm_labels = np.where(ocsvm_outliers == 1, 1, 0)
        ocsvm_labels_all[class_indices] = ocsvm_labels
        y_ocsvm_eval[class_indices[ocsvm_labels == 0]] = 0
        ocsvm_per_class_outliers[i] = np.sum(ocsvm_labels == 0)
        ocsvm_total_outliers += np.sum(ocsvm_labels == 0)
    
    # Skor eşiği ile OCSVM
    ocsvm_scores = best_ocsvm.decision_function(X_scaled)
    threshold = np.percentile(ocsvm_scores, 5)  # %5'lik kesme noktası
    ocsvm_labels_thresholded = np.where(ocsvm_scores < threshold, 0, 1)
    ocsvm_total_outliers_thresholded = np.sum(ocsvm_labels_thresholded == 0)
    
    # OCSVM sonuçlarını yazdır
    print(f"OCSVM (Threshold) Toplam Tespit Edilen Aykırı Değerler: {ocsvm_total_outliers_thresholded}")
    for i in range(10):
        print(f"OCSVM (Threshold) Sınıf {i}'de Tespit Edilen Aykırı Değerler: {np.sum(ocsvm_labels_thresholded[y == i] == 0)}")
    
    ocsvm_accuracy = accuracy_score(y_ocsvm_eval, np.where(ocsvm_labels_thresholded == 0, 0, 1))
    print(f"OCSVM Doğruluk: {ocsvm_accuracy:.4f}")
    
    # 2. LOF Modeli
    lof = create_lof_model()
    lof_scores = -lof.fit_predict(X_scaled)  # Negatif değerler (sklearn: -1=aykırı değer)
    lof_labels = np.where(lof_scores > 0, 1, 0)  # 1=aykırı değer, 0=normal
    
    print(f"LOF Tespit Edilen Aykırı Değer Sayısı: {np.sum(lof_labels == 1)}")
    print(f"LOF Aykırı Değer Yüzdesi: {np.sum(lof_labels == 1) / len(lof_labels) * 100:.2f}%")
    
    y_lof_eval = np.ones_like(y)
    y_lof_eval[lof_labels == 1] = 0
    lof_accuracy = accuracy_score(y_ocsvm_eval, y_lof_eval)
    print(f"LOF Doğruluk: {lof_accuracy:.4f}")
    
    # 3. Isolation Forest Modeli
    iforest = create_isolation_forest(contamination=0.05)
    iforest.fit(X_scaled)
    iforest_scores = iforest.decision_function(X_scaled)
    iforest_outliers = iforest.predict(X_scaled)
    iforest_labels = np.where(iforest_outliers == -1, 0, 1)  # 0=aykırı değer, 1=normal
    
    y_iforest_eval = np.ones_like(y)
    y_iforest_eval[iforest_labels == 0] = 0
    iforest_accuracy = accuracy_score(y_ocsvm_eval, y_iforest_eval)
    print(f"Isolation Forest Doğruluk: {iforest_accuracy:.4f}")
    
    # 4. Elliptic Envelope Modeli
    envelope = create_elliptic_envelope(contamination=0.1)
    envelope_outliers = envelope.fit_predict(X_scaled)
    envelope_labels = np.where(envelope_outliers == -1, 0, 1)  # 0=aykırı değer, 1=normal
    
    envelope_outlier_count = np.sum(envelope_labels == 0)
    envelope_outlier_ratio = envelope_outlier_count / len(X_scaled)
    print(f"Elliptic Envelope Tespit Edilen Aykırı Değerler: {envelope_outlier_count}")
    print(f"Elliptic Envelope Aykırı Değer Oranı: {envelope_outlier_ratio:.2%}")
    
    y_envelope_eval = np.ones_like(y)
    y_envelope_eval[envelope_labels == 0] = 0
    envelope_accuracy = accuracy_score(y_ocsvm_eval, y_envelope_eval)
    print(f"Elliptic Envelope Doğruluk: {envelope_accuracy:.4f}")
    
    # Tüm anomali etiketlerini birleştir
    y_with_outliers = np.copy(y)
    
    # Tüm aykırı değerleri birleştir
    all_outliers = np.union1d(
        np.union1d(
            np.union1d(
                np.where(ocsvm_labels_thresholded == 0)[0], 
                np.where(lof_labels == 1)[0]
            ),
            np.where(iforest_labels == 0)[0]
        ),
        np.where(envelope_labels == 0)[0]
    )
    
    # Aykırı değerlerin sınıfını 10 yap (yeni sınıf)
    y_with_outliers[all_outliers] = 10
    
    print("\n=== Hibrit CNN Modeli Eğitimi Başlıyor ===")
    
    # Orijinal görüntüleri kullan (8x8)
    X_images = digits.images / 16.0  # Normalizasyon
    X_images = X_images.reshape(-1, 8, 8, 1)  # CNN için reshape
    
    # Özel öznitelikleri çıkar
    print("Özel öznitelikler çıkarılıyor...")
    X_custom_features = extract_digit_features(digits.images)
    
    # One-hot encoding - normal sınıflar (0-9) + aykırı değer sınıfı (10)
    y_categorical = to_categorical(y_with_outliers, num_classes=11)
    
    # Veri setini böl
    X_train_img, X_test_img, y_train_cat, y_test_cat = train_test_split(
        X_images, y_categorical, test_size=0.4, random_state=42
    )
    
    # Öznitelik veri setini de aynı şekilde böl
    X_train_features, X_test_features, _, _ = train_test_split(
        X_custom_features, y_categorical, test_size=0.4, random_state=42
    )
    
    # Sınıf ağırlıklarını tanımla
    class_weights = {i: 1.0 for i in range(11)}
    class_weights[1] = 2.5  # 1 rakamı için yüksek ağırlık
    class_weights[2] = 2.0  # 2 rakamı için ağırlık
    class_weights[8] = 2.0  # 8 rakamı için ağırlık
    
    # Hibrit modeli oluştur
    print("Hibrit CNN modeli oluşturuluyor...")
    hybrid_model = create_hybrid_model(
        input_shape=(8, 8, 1), 
        feature_shape=X_custom_features.shape[1]
    )
    
    # Modeli eğit
    print("Model eğitimi başlıyor...")
    history = train_hybrid_model(
        hybrid_model,
        X_train_img, 
        X_train_features, 
        y_train_cat,
        X_test_img, 
        X_test_features, 
        y_test_cat,
        epochs=20,
        batch_size=32,
        class_weights=class_weights
    )
    
    # Eğitim geçmişini görselleştir
    plot_training_history(history)
    
    # Hibrit modeli değerlendir
    print("\n=== Model Değerlendirmesi ===")
    test_loss, test_accuracy = hybrid_model.evaluate(
        [X_test_img, X_test_features], y_test_cat, verbose=1
    )
    
    # Modelin tahminlerini al
    y_pred_proba = hybrid_model.predict([X_test_img, X_test_features])
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    # Her rakam için doğruluk analizini yap
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
    
    # Karmaşıklık matrisi
    plot_confusion_matrix(y_true, y_pred_classes)
    
    # CNN tarafından tespit edilen aykırı değerleri bul
    cnn_outliers = np.where(y_pred_classes == 10)[0]
    true_outliers = np.where(y_true == 10)[0]
    
    # Her bir model için sadece o model tarafından tespit edilen aykırı değerler
    # Test indekslerine göre dönüştür
    test_indices = np.arange(len(y_test_cat))
    
    cnn_only_outliers = test_indices[np.setdiff1d(np.where(y_pred_classes == 10)[0], np.where(y_true == 10)[0])]
    
    # Her bir model için aykırı değer tespiti sonuçlarını tut
    ocsvm_outliers = np.where(ocsvm_labels_thresholded == 0)[0]
    lof_outliers = np.where(lof_labels == 1)[0]
    iforest_outliers = np.where(iforest_labels == 0)[0]
    envelope_outliers = np.where(envelope_labels == 0)[0]
    
    # Sonuçları yazdır
    print("\n=== Model Metrikleri ===")
    print(f"CNN Test Doğruluğu: {test_accuracy:.4f}")
    print(f"CNN Precision: {cnn_precision:.4f}")
    print(f"CNN Recall: {cnn_recall:.4f}")
    print(f"CNN F1-Score: {cnn_f1:.4f}")
    print(f"CNN Normal Rakamlar Doğruluğu: {normal_accuracy:.4f}")
    print(f"CNN Aykırı Değer Tespit Doğruluğu: {outlier_accuracy:.4f}")
    print(f"CNN Aykırı Değer Precision: {outlier_precision:.4f}")
    print(f"CNN Aykırı Değer Recall: {outlier_recall:.4f}")
    
    print("\n=== Aykırı Değer Tespit Sonuçları ===")
    print(f"OCSVM Tespit Edilen Toplam Aykırı Değer: {len(ocsvm_outliers)}")
    print(f"LOF Tespit Edilen Aykırı Değer: {len(lof_outliers)}")
    print(f"Isolation Forest Tespit Edilen Aykırı Değer: {len(iforest_outliers)}")
    print(f"Elliptic Envelope Tespit Edilen Aykırı Değer: {len(envelope_outliers)}")
    print(f"CNN Tespit Edilen Aykırı Değer: {len(cnn_outliers)}")
    print(f"CNN'in Sadece Tespit Ettiği Aykırı Değerler: {len(cnn_only_outliers)}")
    
    # Her rakam için yanlış sınıflandırmaları görselleştir
    for i in range(10):
        visualize_misclassified_digits(X_test_img, y_true, y_pred_classes, i)
    
    # Aykırı değer tespit sonuçlarını görselleştir
    model_results = {
        'ocsvm': ocsvm_outliers,
        'lof': lof_outliers,
        'iforest': iforest_outliers,
        'envelope': envelope_outliers,
        'cnn': cnn_outliers
    }
    
    plot_outlier_detection_results(X_pca, y_with_outliers, model_results)


if __name__ == "__main__":
    main()