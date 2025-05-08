# utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


def extract_digit_features(images, reshape=True):
    """
    Rakam görüntülerinden öznitelik çıkarma
    
    Args:
        images: Görüntü veri seti
        reshape: Yeniden boyutlandırma yapılacak mı
        
    Returns:
        Çıkarılan öznitelikler
    """
    features = []
    
    for img in images:
        if reshape:
            img = img.reshape(8, 8)
        img = img.squeeze()  # Tek kanallı görüntü
        
        # Temel öznitelikler
        horizontal_lines = np.sum(img > 0.5, axis=1)  # Satır başına aktif piksel
        vertical_lines = np.sum(img > 0.5, axis=0)    # Sütun başına aktif piksel
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
        
        # 1 VE 8 RAKAMLARI ARASINDA AYRIM İÇİN YENİ ÖZNİTELİKLER
        
        # 1. En-boy oranı - rakam 1 uzun ve dar
        height = np.sum(np.any(img > 0.5, axis=1))
        width = np.sum(np.any(img > 0.5, axis=0))
        aspect_ratio = height / (width + 1e-6)  # Sıfıra bölmeyi önle
        
        # 2. Dikey simetri - rakam 8 daha simetrik olmalı
        left_half = img[:, :4]
        right_half = img[:, 4:]
        right_half_flipped = np.fliplr(right_half)
        vertical_symmetry = np.mean(np.abs(left_half - right_half_flipped))
        
        # 3. Merkez sütun yoğunluğu - rakam 1 orta sütunlarda yüksek yoğunluğa sahip
        center_column_density = np.mean(img[:, 3:5])
        
        # 4. Rakam 1'e özgü öznitelik - tek dikey çizgi kontrolü
        vertical_line_check = np.mean([np.sum(img[i, 3:5] > 0.5) > 0 for i in range(8)])
        
        # Tüm öznitelikleri birleştir
        feature_vector = np.concatenate([
            horizontal_lines, vertical_lines,
            [center_region, top_region, middle_region, bottom_region],
            [top_left, top_right, bottom_left, bottom_right],
            [upper_loop, lower_loop, left_edge, right_edge, center_line],
            # 1 ve 8 ayrımı için yeni öznitelikler
            [aspect_ratio, vertical_symmetry, center_column_density, vertical_line_check]
        ])
        
        features.append(feature_vector)
    
    return np.array(features)


def analyze_per_digit_accuracy(y_true, y_pred):
    """
    Her rakam sınıfı için doğruluk analizi yapar
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    """
    print("\n===== Rakam Başına Doğruluk Analizi =====")
    for digit in range(10):
        idx = np.where(y_true == digit)[0]
        if len(idx) > 0:
            accuracy = np.mean(y_pred[idx] == y_true[idx])
            print(f"Rakam {digit} için doğruluk: {accuracy:.4f}")


def visualize_misclassified_digits(X_test, y_true, y_pred, digit_class):
    """
    Yanlış sınıflandırılan rakamları görselleştirir
    
    Args:
        X_test: Test görüntüleri
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        digit_class: Görselleştirilecek rakam sınıfı
    """
    # Belirli bir rakam sınıfını al
    class_indices = np.where(y_true == digit_class)[0]
    # Hatalı sınıflandırılanları bul
    misclassified = class_indices[y_pred[class_indices] != digit_class]
    
    if len(misclassified) == 0:
        print(f"Rakam {digit_class} için yanlış sınıflandırılmış örnek bulunamadı")
        return
    
    # En fazla 10 tane göster
    n_samples = min(10, len(misclassified))
    plt.figure(figsize=(15, 6))
    
    for i in range(n_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[misclassified[i]].reshape(8, 8), cmap='gray')
        plt.title(f"True: {digit_class}, Pred: {y_pred[misclassified[i]]}")
        plt.axis('off')
    
    plt.suptitle(f"Yanlış Sınıflandırılmış Rakam {digit_class} Örnekleri")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Eğitim geçmişini görselleştirir
    
    Args:
        history: Keras model eğitim geçmişi
    """
    plt.figure(figsize=(12, 5))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], color='b', label="Doğrulama Kaybı")
    plt.plot(history.history['loss'], color='r', label="Eğitim Kaybı")
    plt.title("Model Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.legend()
    
    # Accuracy grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], color='b', label="Doğrulama Doğruluğu")
    plt.plot(history.history['accuracy'], color='r', label="Eğitim Doğruluğu")
    plt.title("Model Doğruluğu")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Karmaşıklık matrisini görselleştirir
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    """
    confusion_mtx = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
    plt.xlabel("Tahmin Edilen Etiket")
    plt.ylabel("Gerçek Etiket")
    plt.title("Karmaşıklık Matrisi")
    plt.tight_layout()
    plt.show()


def plot_outlier_detection_results(X_pca, y_with_outliers, model_results):
    """
    Aykırı değer tespiti sonuçlarını görselleştirir
    
    Args:
        X_pca: PCA ile boyut indirgenmiş veriler
        y_with_outliers: Aykırı değer içeren etiketler
        model_results: Modellerin aykırı değer tespiti sonuçları
    """
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    jitter = np.random.normal(0, 0.02, size=X_pca.shape)
    normal_indices = np.where(y_with_outliers != 10)[0]
    
    # Normal verileri çiz
    ax.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], 
               c='black', s=20, label='Normal', alpha=0.7)
    
    # Her bir model için aykırı değerleri çiz
    for model_name, outlier_indices in model_results.items():
        if model_name == 'ocsvm':
            color = 'red'
        elif model_name == 'lof':
            color = 'orange'
        elif model_name == 'iforest':
            color = 'blue'
        elif model_name == 'envelope':
            color = 'green'
        elif model_name == 'cnn':
            color = 'purple'
        else:
            color = 'gray'
            
        ax.scatter(X_pca[outlier_indices, 0] + jitter[outlier_indices, 0],
                   X_pca[outlier_indices, 1] + jitter[outlier_indices, 1],
                   c=color, s=30, label=f'{model_name} Aykırı Değerler', alpha=0.8)
    
    # Grafik özellikleri
    ax.set_xlabel("PCA Bileşeni 1", fontsize=14)
    ax.set_ylabel("PCA Bileşeni 2", fontsize=14)
    ax.set_title("Aykırı Değer Tespiti Sonuçları", fontsize=16)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)
    
    # Sonuçları göster
    model_stats = "\n".join([f"{model}: {len(indices)} aykırı değer" 
                            for model, indices in model_results.items()])
    plt.figtext(0.5, 0.01, model_stats, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()