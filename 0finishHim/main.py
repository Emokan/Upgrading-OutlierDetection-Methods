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

# Import custom modules
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
    # Suppress warnings
    warnings.filterwarnings("ignore", message="The covariance matrix associated to your dataset is not full rank")
    
    print("=== MNIST Digit Dataset Outlier Detection and Classification ===")
    
    # Load the dataset
    print("Loading the dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Prepare anomaly detection models
    print("Preparing anomaly detection models...")
    
    # 1. OCSVM Model
    ocsvm_grid = create_ocsvm_model()
    ocsvm_grid.fit(X_scaled)
    best_ocsvm = ocsvm_grid.best_estimator_
    print(f"Best OCSVM parameters: {ocsvm_grid.best_params_}")
    
    # Perform outlier detection with OCSVM for each class
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
    
    # OCSVM with score threshold
    ocsvm_scores = best_ocsvm.decision_function(X_scaled)
    threshold = np.percentile(ocsvm_scores, 5)  # 5% cutoff point
    ocsvm_labels_thresholded = np.where(ocsvm_scores < threshold, 0, 1)
    ocsvm_total_outliers_thresholded = np.sum(ocsvm_labels_thresholded == 0)
    
    # Print OCSVM results
    print(f"OCSVM (Threshold) Total Detected Outliers: {ocsvm_total_outliers_thresholded}")
    for i in range(10):
        print(f"OCSVM (Threshold) Detected Outliers in Class {i}: {np.sum(ocsvm_labels_thresholded[y == i] == 0)}")
    
    ocsvm_accuracy = accuracy_score(y_ocsvm_eval, np.where(ocsvm_labels_thresholded == 0, 0, 1))
    print(f"OCSVM Accuracy: {ocsvm_accuracy:.4f}")
    
    # 2. LOF Model
    lof = create_lof_model()
    lof_scores = -lof.fit_predict(X_scaled)  # Negative values (sklearn: -1=outlier)
    lof_labels = np.where(lof_scores > 0, 1, 0)  # 1=outlier, 0=normal
    
    print(f"LOF Detected Outlier Count: {np.sum(lof_labels == 1)}")
    print(f"LOF Outlier Percentage: {np.sum(lof_labels == 1) / len(lof_labels) * 100:.2f}%")
    
    y_lof_eval = np.ones_like(y)
    y_lof_eval[lof_labels == 1] = 0
    lof_accuracy = accuracy_score(y_ocsvm_eval, y_lof_eval)
    print(f"LOF Accuracy: {lof_accuracy:.4f}")
    
    # 3. Isolation Forest Model
    iforest = create_isolation_forest(contamination=0.05)
    iforest.fit(X_scaled)
    iforest_scores = iforest.decision_function(X_scaled)
    iforest_outliers = iforest.predict(X_scaled)
    iforest_labels = np.where(iforest_outliers == -1, 0, 1)  # 0=outlier, 1=normal
    
    y_iforest_eval = np.ones_like(y)
    y_iforest_eval[iforest_labels == 0] = 0
    iforest_accuracy = accuracy_score(y_ocsvm_eval, y_iforest_eval)
    print(f"Isolation Forest Accuracy: {iforest_accuracy:.4f}")
    
    # 4. Elliptic Envelope Model
    envelope = create_elliptic_envelope(contamination=0.1)
    envelope_outliers = envelope.fit_predict(X_scaled)
    envelope_labels = np.where(envelope_outliers == -1, 0, 1)  # 0=outlier, 1=normal
    
    envelope_outlier_count = np.sum(envelope_labels == 0)
    envelope_outlier_ratio = envelope_outlier_count / len(X_scaled)
    print(f"Elliptic Envelope Detected Outliers: {envelope_outlier_count}")
    print(f"Elliptic Envelope Outlier Ratio: {envelope_outlier_ratio:.2%}")
    
    y_envelope_eval = np.ones_like(y)
    y_envelope_eval[envelope_labels == 0] = 0
    envelope_accuracy = accuracy_score(y_ocsvm_eval, y_envelope_eval)
    print(f"Elliptic Envelope Accuracy: {envelope_accuracy:.4f}")
    
    # Combine all anomaly labels
    y_with_outliers = np.copy(y)
    
    # Merge all outliers
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
    
    # Assign a new class (10) to outliers
    y_with_outliers[all_outliers] = 10
    
    print("\n=== Hybrid CNN Model Training Starts ===")
    
    # Use original images (8x8)
    X_images = digits.images / 16.0  # Normalization
    X_images = X_images.reshape(-1, 8, 8, 1)  # Reshape for CNN
    
    # Extract custom features
    print("Extracting custom features...")
    X_custom_features = extract_digit_features(digits.images)
    
    # One-hot encoding - normal classes (0-9) + outlier class (10)
    y_categorical = to_categorical(y_with_outliers, num_classes=11)
    
    # Split the dataset
    X_train_img, X_test_img, y_train_cat, y_test_cat = train_test_split(
        X_images, y_categorical, test_size=0.4, random_state=42
    )
    
    # Split the feature dataset similarly
    X_train_features, X_test_features, _, _ = train_test_split(
        X_custom_features, y_categorical, test_size=0.4, random_state=42
    )
    
    # Define class weights
    class_weights = {i: 1.0 for i in range(11)}
    class_weights[1] = 2.5  # Higher weight for digit 1
    class_weights[2] = 2.0  # Weight for digit 2
    class_weights[8] = 2.0  # Weight for digit 8
    
    # Create the hybrid model
    print("Creating the hybrid CNN model...")
    hybrid_model = create_hybrid_model(
        input_shape=(8, 8, 1), 
        feature_shape=X_custom_features.shape[1]
    )
    
    # Train the model
    print("Starting model training...")
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
    
    # Visualize training history
    plot_training_history(history)
    
    # Evaluate the hybrid model
    print("\n=== Model Evaluation ===")
    test_loss, test_accuracy = hybrid_model.evaluate(
        [X_test_img, X_test_features], y_test_cat, verbose=1
    )
    
    # Get model predictions
    y_pred_proba = hybrid_model.predict([X_test_img, X_test_features])
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    
    # Analyze accuracy for each digit
    analyze_per_digit_accuracy(y_true, y_pred_classes)
    
    # Metrics for all classes
    cnn_precision = precision_score(y_true, y_pred_classes, average='weighted')
    cnn_recall = recall_score(y_true, y_pred_classes, average='weighted')
    cnn_f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    # Segmented accuracy metrics
    normal_indices = np.where(y_true != 10)[0]
    normal_accuracy = accuracy_score(y_true[normal_indices], y_pred_classes[normal_indices])
    
    # Accuracy for outliers (class 10 only)
    outlier_indices = np.where(y_true == 10)[0]
    if len(outlier_indices) > 0:
        outlier_accuracy = accuracy_score(y_true[outlier_indices], y_pred_classes[outlier_indices])
    else:
        outlier_accuracy = 0.0
    
    # Precision and recall for outlier detection
    outlier_precision = precision_score(y_true == 10, y_pred_classes == 10)
    outlier_recall = recall_score(y_true == 10, y_pred_classes == 10)
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes)
    
    # Find outliers detected by CNN
    cnn_outliers = np.where(y_pred_classes == 10)[0]
    true_outliers = np.where(y_true == 10)[0]
    
    # Outliers detected only by each model
    # Convert to test indices
    test_indices = np.arange(len(y_test_cat))
    
    cnn_only_outliers = test_indices[np.setdiff1d(np.where(y_pred_classes == 10)[0], np.where(y_true == 10)[0])]
    
    # Store outlier detection results for each model
    ocsvm_outliers = np.where(ocsvm_labels_thresholded == 0)[0]
    lof_outliers = np.where(lof_labels == 1)[0]
    iforest_outliers = np.where(iforest_labels == 0)[0]
    envelope_outliers = np.where(envelope_labels == 0)[0]
    
    # Print results
    print("\n=== Model Metrics ===")
    print(f"CNN Test Accuracy: {test_accuracy:.4f}")
    print(f"CNN Precision: {cnn_precision:.4f}")
    print(f"CNN Recall: {cnn_recall:.4f}")
    print(f"CNN F1-Score: {cnn_f1:.4f}")
    print(f"CNN Normal Digits Accuracy: {normal_accuracy:.4f}")
    print(f"CNN Outlier Detection Accuracy: {outlier_accuracy:.4f}")
    print(f"CNN Outlier Precision: {outlier_precision:.4f}")
    print(f"CNN Outlier Recall: {outlier_recall:.4f}")
    
    print("\n=== Outlier Detection Results  ===")
    print(f"OCSVM Total Detected Outliers: {len(ocsvm_outliers)}")
    print(f"LOF Detected Outliers: {len(lof_outliers)}")
    print(f"Isolation Forest Detected Outliers: {len(iforest_outliers)}")
    print(f"Elliptic Envelope Detected Outliers: {len(envelope_outliers)}")
    print(f"CNN Detected Outliers: {len(cnn_outliers)}")
    print(f"CNN Only Outliers: {len(cnn_only_outliers)}")
    
    # Visualize misclassifications for each digit
    for i in range(10):
        visualize_misclassified_digits(X_test_img, y_true, y_pred_classes, i)
    
    # Visualize outlier detection results
    model_results = {
        'OCSVM Detected Outliers': ocsvm_outliers,
        'LOF Detected Outliers': lof_outliers,
        'Isolation Forest Detected Outliers': iforest_outliers,
        'Elliptic Envelope Outliers': envelope_outliers,
        'CNN Outliers': cnn_outliers
    }
    
    plot_outlier_detection_results(X_pca, y_with_outliers, model_results)


if __name__ == "__main__":
    main()
