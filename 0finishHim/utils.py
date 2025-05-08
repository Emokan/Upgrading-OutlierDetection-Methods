# utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

def extract_digit_features(images, reshape=True):
    features = []
    
    for img in images:
        if reshape:
            img = img.reshape(8, 8)
        img = img.squeeze()  # Single-channel image
        
        # Basic features
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
        
        # NEW FEATURES FOR DISTINGUISHING BETWEEN DIGITS 1 AND 8
        
        # 1. Aspect ratio - digit 1 is tall and narrow
        height = np.sum(np.any(img > 0.5, axis=1))
        width = np.sum(np.any(img > 0.5, axis=0))
        aspect_ratio = height / (width + 1e-6)  # Prevent division by zero
        
        # 2. Vertical symmetry - digit 8 should be more symmetric
        left_half = img[:, :4]
        right_half = img[:, 4:]
        right_half_flipped = np.fliplr(right_half)
        vertical_symmetry = np.mean(np.abs(left_half - right_half_flipped))
        
        # 3. Center column density - digit 1 has high density in center columns
        center_column_density = np.mean(img[:, 3:5])
        
        # 4. Feature specific to digit 1 - single vertical line check
        vertical_line_check = np.mean([np.sum(img[i, 3:5] > 0.5) > 0 for i in range(8)])
        
        # Combine all features
        feature_vector = np.concatenate([
            horizontal_lines, vertical_lines,
            [center_region, top_region, middle_region, bottom_region],
            [top_left, top_right, bottom_left, bottom_right],
            [upper_loop, lower_loop, left_edge, right_edge, center_line],
            # New features for distinguishing 1 and 8
            [aspect_ratio, vertical_symmetry, center_column_density, vertical_line_check]
        ])
        
        features.append(feature_vector)
    
    return np.array(features)


def analyze_per_digit_accuracy(y_true, y_pred):
    print("\n===== Per-Digit Accuracy Analysis =====")
    for digit in range(10):
        idx = np.where(y_true == digit)[0]
        if len(idx) > 0:
            accuracy = np.mean(y_pred[idx] == y_true[idx])
            print(f"Accuracy for digit {digit}: {accuracy:.4f}")


def visualize_misclassified_digits(X_test, y_true, y_pred, digit_class):
    # Get a specific digit class
    class_indices = np.where(y_true == digit_class)[0]
    # Find misclassified samples
    misclassified = class_indices[y_pred[class_indices] != digit_class]
    
    if len(misclassified) == 0:
        print(f"No misclassified samples found for digit {digit_class}")
        return
    
    # Show up to 10 samples
    n_samples = min(10, len(misclassified))
    plt.figure(figsize=(13, 5))
    
    for i in range(n_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[misclassified[i]].reshape(8, 8), cmap='gray')
        plt.title(f"True: {digit_class}, Pred: {y_pred[misclassified[i]]}")
        plt.axis('off')
    
    plt.suptitle(f"Misclassified Digit {digit_class} Samples")
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], color='b', label="Validation Loss")
    plt.plot(history.history['loss'], color='r', label="Training Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], color='b', label="Validation Accuracy")
    plt.plot(history.history['accuracy'], color='r', label="Training Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    confusion_mtx = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_outlier_detection_results(X_pca, y_with_outliers, model_results):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    jitter = np.random.normal(0, 0.02, size=X_pca.shape)
    normal_indices = np.where(y_with_outliers != 10)[0]
    
    # Plot normal data
    ax.scatter(X_pca[normal_indices, 0], X_pca[normal_indices, 1], 
               c='black', s=20, label='Normal', alpha=0.7)
    
    # Plot outliers for each model
    for model_name, outlier_indices in model_results.items():
        if model_name == 'OCSVM Detected Outliers':
            color = 'red'
        elif model_name == 'LOF Detected Outliers':
            color = 'orange'
        elif model_name == 'Isolation Forest Detected Outliers':
            color = 'blue'
        elif model_name == 'Elliptic Envelope Outliers':
            color = 'green'
        elif model_name == 'CNN Outliers':
            color = 'purple'
        else:
            color = 'gray'
            
        ax.scatter(X_pca[outlier_indices, 0] + jitter[outlier_indices, 0],
                   X_pca[outlier_indices, 1] + jitter[outlier_indices, 1],
                   c=color, s=30, label=f'{model_name} Outliers', alpha=0.8)
    
    # Graph properties
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.set_title("Outlier Detection Results", fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.8)
    
    # Show results
    model_stats = "\n".join([f"{model}: {len(indices)} outliers" 
                            for model, indices in model_results.items()])
    
    plt.figtext(0.5, 0.05, model_stats, ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show()
