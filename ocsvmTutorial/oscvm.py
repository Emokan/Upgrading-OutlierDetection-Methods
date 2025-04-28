# https://www.geeksforgeeks.org/support-vector-machine-svm-for-anomaly-detection/

# Synthetic dataset
from sklearn.datasets import make_classification
#Data processing
import pandas as pd
import numpy as np
from collections import Counter 
#Visualization
import matplotlib.pyplot as plt 
#Model and performance
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

# Create an imbalanced dataset
X, y = make_classification(n_samples=100000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2, 
                           n_clusters_per_class=1,
                           weights=[0.995,0.005],
                           class_sep=0.5, random_state=0)

#Convert the data from numpy array to a pandas dataframe
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

#Check the target distribution
df['target'].value_counts(normalize = True)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Check the number of records
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])

# Analyze class distribution in the training set
class_counts = Counter(y_train)
majority_class, majority_count = sorted(class_counts.items())[0]
minority_class, minority_count = sorted(class_counts.items())[-1]

print(f"The training dataset has {majority_count} records for the majority class ({majority_class})")
print(f"and {minority_count} records for the minority class ({minority_class})")

# Train the one support vector machine (SVM) model
one_class_svm = OneClassSVM(nu = 0.01, kernel = 'rbf', gamma = 'auto').fit(X_train) 

# Predict the anomalies 
prediction = one_class_svm.predict(X_test)

#Change the anomalies' values and to make it consistent with the true values
prediction = [1 if i==-1 else 0 for i in prediction]

#Check the model performance
print(classification_report(y_test, prediction))

# Get the scores for the testing dataset
score = one_class_svm.score_samples(X_test)

#Check the score for 2% of outliers
score_threshold = np.percentile(score, 2)
print(f'The customized score threshold for 2% of outliers is {score_threshold: .2f}')
# Check the model performance at 2% threshold 
customized_prediction = [1 if i < score_threshold else 0 for i in score]
#Check the prediction performance 
print(classification_report(y_test, customized_prediction))

# Put the testing dataset and predictions in the same dataframe
df_test = pd.DataFrame(X_test, columns=['feature1', 'feature2'])
df_test['y_test'] = y_test
df_test['one_class_svm_prediction'] = prediction
df_test['one_class_svm_prediction_customized'] = customized_prediction

# Visualize the actual and predicted anomalies
fig, (ax0, ax1, ax2)=plt.subplots(1,3, sharey=True, figsize=(20,6))

#Ground truth
ax0.set_title('Original')
ax0.scatter(df_test['feature1'], df_test['feature2'], c=df_test['y_test'], cmap='rainbow')

#One-Class SVM Predictions
ax1.set_title('One-Class SVM Predictions')
ax1.scatter(df_test['feature1'], df_test['feature2'], c=df_test['one_class_svm_prediction'], cmap='rainbow')

#One-Class SVM Predictions With Customized Threshold
ax2.set_title('One-Class SVM Predictions With Customized Threshold')
ax2.scatter(df_test['feature1'], df_test['feature2'], c=df_test['one_class_svm_prediction_customized'], cmap='rainbow')

plt.show()