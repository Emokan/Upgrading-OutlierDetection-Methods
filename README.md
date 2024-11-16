# Upgrading-OutlierDetection-Methods
One-Class Support Vector Machine (OCSVM) 

Project Overview

This project is inspired by the paper titled “Two-Stage Approach With Combination of Outlier Detection Method and Deep Learning Enhances Automatic Epileptic Seizure Detection.” Our goal is to enhance the One-Class Support Vector Machine (OCSVM) method for anomaly detection by incorporating various outlier detection techniques and exploring their combined impact.

We used the breast cancer dataset available in the sklearn library as a test case for evaluating the performance of different anomaly detection techniques.

Key Features:

	•	Dataset: The breast_cancer dataset from sklearn is used for training and testing. This dataset contains features of breast cancer cell nuclei, which are used for classifying malignant vs. benign tumors.
	•	Outlier Detection Methods: The following techniques are explored:
	•	Isolation Forest (IF)
	•	Local Outlier Factor (LOF)
	•	One-Class SVM (OCSVM)
	•	Interquartile Range (IQR)
	•	Z-Score
These techniques are applied to detect anomalies in the dataset.
	•	Combination of Outlier Detection Methods: Anomalies are identified using the above methods individually, and their combination is used to pinpoint outliers that are detected by multiple methods. We also examine common outliers detected by all methods.
	•	Evaluation: The performance of these methods is evaluated using precision, recall, and F1-score, with each method’s ability to detect outliers compared. Metrics are calculated using sklearn.metrics.

Workflow:

	1.	Preprocessing:
	•	Data is standardized using StandardScaler.
	•	Feature selection techniques like PCA and Recursive Feature Elimination (RFE) are applied for dimensionality reduction and feature selection.
	2.	Outlier Detection:
	•	Isolation Forest, LOF, OCSVM, IQR, and Z-score methods are used for detecting outliers.
	3.	Visualization:
	•	The results of the outlier detection methods are visualized using 2D scatter plots.
	4.	Metrics Calculation:
	•	Precision, Recall, and F1-Score are calculated for each method’s predictions.

Clone the repository:
 git clone https://github.com/birkasiyer/Upgrading-OutlierDetection_Methods.git
 


 References:

	•	“Two-Stage Approach With Combination of Outlier Detection Method and Deep Learning Enhances Automatic Epileptic Seizure Detection”. This paper inspired the incorporation of outlier detection methods into the automatic epileptic seizure detection process.
