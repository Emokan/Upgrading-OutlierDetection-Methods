# https://www.kaggle.com/code/sid3945/detecting-outliers-elliptical-envelope

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

data_dir = "/Users/dogu/Desktop/graduated/eEnvelopeTutorial/gearbox-fault-diagnosis"

for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Healthy gearbox
# ---------------
h30hz0  = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz0.csv"))
h30hz10 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz10.csv"))
h30hz20 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz20.csv"))
h30hz30 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz30.csv"))
h30hz40 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz40.csv"))
h30hz50 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz50.csv"))
h30hz60 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz60.csv"))
h30hz70 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz70.csv"))
h30hz80 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz80.csv"))
h30hz90 = pd.read_csv(os.path.join(data_dir, "Healthy", "h30hz90.csv"))
# Broken gearbox
# --------------
b30hz0  = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz0.csv"))
b30hz10 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz10.csv"))
b30hz20 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz20.csv"))
b30hz30 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz30.csv"))
b30hz40 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz40.csv"))
b30hz50 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz50.csv"))
b30hz60 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz60.csv"))
b30hz70 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz70.csv"))
b30hz80 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz80.csv"))
b30hz90 = pd.read_csv(os.path.join(data_dir, "BrokenTooth", "b30hz90.csv"))

failure = 0

load = 0

h30hz0['load'] = load*np.ones((len(h30hz0.index),1))
failureArray = np.zeros((len(h30hz0.index),1))
h30hz0['failure'] = failureArray

load = 10

h30hz10['load'] = load*np.ones((len(h30hz10.index),1))
failureArray = np.zeros((len(h30hz10.index),1))
h30hz10['failure'] = failureArray

load = 20

h30hz20['load'] = load*np.ones((len(h30hz20.index),1))
failureArray = np.zeros((len(h30hz20.index),1))
h30hz20['failure'] = failureArray

load = 30

h30hz30['load'] = load*np.ones((len(h30hz30.index),1))
failureArray = np.zeros((len(h30hz30.index),1))
h30hz30['failure'] = failureArray

load = 40

h30hz40['load'] = load*np.ones((len(h30hz40.index),1))
failureArray = np.zeros((len(h30hz40.index),1))
h30hz40['failure'] = failureArray

load = 50

h30hz50['load'] = load*np.ones((len(h30hz50.index),1))
failureArray = np.zeros((len(h30hz50.index),1))
h30hz50['failure'] = failureArray

load = 60

h30hz60['load'] = load*np.ones((len(h30hz60.index),1))
failureArray = np.zeros((len(h30hz60.index),1))
h30hz60['failure'] = failureArray

load = 70

h30hz70['load'] = load*np.ones((len(h30hz70.index),1))
failureArray = np.zeros((len(h30hz70.index),1))
h30hz70['failure'] = failureArray

load = 80

h30hz80['load'] = load*np.ones((len(h30hz80.index),1))
failureArray = np.zeros((len(h30hz80.index),1))
h30hz80['failure'] = failureArray

load = 90

h30hz90['load'] = load*np.ones((len(h30hz90.index),1))
failureArray = np.zeros((len(h30hz90.index),1))
h30hz90['failure'] = failureArray

failure = 1

load = 0

b30hz0['load'] = load*np.ones((len(b30hz0.index),1))
failureArray = np.ones((len(b30hz0.index),1))
b30hz0['failure'] = failureArray

load = 10 

b30hz10['load'] = load*np.ones((len(b30hz10.index),1))
failureArray = np.ones((len(b30hz10.index),1))
b30hz10['failure'] = failureArray

load = 20 

b30hz20['load'] = load*np.ones((len(b30hz20.index),1))
failureArray = np.ones((len(b30hz20.index),1))
b30hz20['failure'] = failureArray

load = 30 

b30hz30['load'] = load*np.ones((len(b30hz30.index),1))
failureArray = np.ones((len(b30hz30.index),1))
b30hz30['failure'] = failureArray

load = 40 

b30hz40['load'] = load*np.ones((len(b30hz40.index),1))
failureArray = np.ones((len(b30hz40.index),1))
b30hz40['failure'] = failureArray

load = 50 

b30hz50['load'] = load*np.ones((len(b30hz50.index),1))
failureArray = np.ones((len(b30hz50.index),1))
b30hz50['failure'] = failureArray

load = 60 

b30hz60['load'] = load*np.ones((len(b30hz60.index),1))
failureArray = np.ones((len(b30hz60.index),1))
b30hz60['failure'] = failureArray

load = 70 

b30hz70['load'] = load*np.ones((len(b30hz70.index),1))
failureArray = np.ones((len(b30hz70.index),1))
b30hz70['failure'] = failureArray

load = 80 

b30hz80['load'] = load*np.ones((len(b30hz80.index),1))
failureArray = np.ones((len(b30hz80.index),1))
b30hz80['failure'] = failureArray

load = 90 

b30hz90['load'] = load*np.ones((len(b30hz90.index),1))
failureArray = np.ones((len(b30hz90.index),1))
b30hz90['failure'] = failureArray

# Broken 

broken_df = pd.concat([b30hz0,b30hz10,b30hz20,b30hz30,b30hz40,b30hz50,b30hz60,b30hz70,b30hz80,b30hz90],axis=0,ignore_index=True)

# Healthy

healthy_df = pd.concat([h30hz0,h30hz10,h30hz20,h30hz30,h30hz40,h30hz50,h30hz60,h30hz70,h30hz80,h30hz90],axis=0,ignore_index=True)

# Finally the aggregated dataset is below
gear_data   = pd.concat([broken_df,healthy_df], axis =0)

training_features = ['a1', 'a2', 'a3', 'a4']
label = ['failure']
X = gear_data[training_features]
y=gear_data[label]

X,y = shuffle(X,y)
X.head()

sns.set_style("darkgrid")
sns.scatterplot(x=X['a1'][:], y=X['a2'][:])
plt.show()

sns.set_style("darkgrid")
sns.scatterplot(x=X['a2'][:], y=X['a3'][:])
plt.show()

sns.set_style("darkgrid")
sns.scatterplot(x=X['a3'][:], y=X['a4'][:])
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,3))
sns.histplot(X['a1'][:], ax=ax[0], color="darkblue", kde=True)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,3))
sns.histplot(X['a2'][:], ax=ax[0], color="darkblue", kde=True)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,3))
sns.histplot(X['a3'][:], ax=ax[0], color="darkblue", kde=True)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,3))
sns.histplot(X['a4'][:], ax=ax[0], color="darkblue", kde=True)

elpenv = EllipticEnvelope(contamination=0.1, random_state=2)

# Returns 1 for Inliers and -1 for outliers
pred = elpenv.fit_predict(X.iloc[:500]) #this fits he model and predicts in a single command works same as fit X and predict X

outlier_index = np.where(pred==-1)
outlier_index  
outlier_values = X.iloc[outlier_index]

sns.scatterplot(x=X['a1'][:500], y=X['a2'][:500], color = 'b')
sns.scatterplot(x=outlier_values['a1'][:500],y=outlier_values['a2'][:500], color='r')
plt.show()

# Toplam veri sayısı ve bulunan outlier sayısını yazdır
total_samples = len(X.iloc[:500])
outlier_count = len(outlier_index[0])
print(f"Total Data: {total_samples}")
print(f"Detected Outliers: {outlier_count}")
print(f"Outlier Ratio: {outlier_count / total_samples:.2%}")