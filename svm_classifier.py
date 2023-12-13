import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE

raw_df= pd.read_csv("../data-final.csv", sep='\t', nrows=1000)
dataset=raw_df.copy()

dataset.drop(dataset.columns[50:107], axis=1, inplace=True)
dataset.drop(dataset.columns[51:], axis=1, inplace=True)

print('Is there any missing value? ', dataset.isnull().values.any())
print('How many missing values? ', dataset.isnull().values.sum())
dataset.dropna(inplace=True)
print('Number of participants after eliminating missing values: ', len(dataset))

positive = [  'EXT1','EXT3','EXT5','EXT7','EXT9',
              'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10',
              'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',
              'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',
              'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10', ]

negative = [ 'EXT2','EXT4','EXT6','EXT8','EXT10',
             'EST2','EST4',
             'AGR1','AGR3','AGR5','AGR7',
             'CSN2','CSN4','CSN6','CSN8',
             'OPN2','OPN4','OPN6', ]

dataset[positive] = dataset[positive].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
dataset[negative] = dataset[negative].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})
cols = positive + negative
database = dataset[sorted(cols) + ['country']]

EXT = list(database.columns[:10])
EST = list(database.columns[10:20])
AGR = list(database.columns[20:30])
CSN = list(database.columns[30:40])
OPN = list(database.columns[40:50])

dimensions = [EXT,EST,AGR,CSN,OPN]
dimension_averages=["extraversion","neuroticism","agreeableness","conscientiousness","openness"]

for d in range(len(dimensions)):
    dataset[dimension_averages[d]] = dataset[dimensions[d]].mean(axis=1)

dataset['nation'] = dataset['country'].apply(lambda x: 1 if x =='US' else (2 if x =='GB' else (3 if x=='CA' else (4 if x=='AU' else 0))))
dataset.drop("country", axis=1, inplace=True)

dataset = dataset[dataset['nation'] != 0]
dataset['nation'] = dataset['nation'].map({1: 0, 2: 1, 3: 2, 4: 3})

dataset = dataset.sample(n=len(dataset),random_state=42)
y = dataset['nation']
X = dataset.drop('nation',axis=1)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X,y)
print('Original data size:\n',y.value_counts())
print('Data size after SMOTE is done:\n',y_smote.value_counts())

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.1, random_state=42)
'''
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Assuming X_train and X_test are already defined

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM classifier
svm_model = SVC()

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = svm_model.predict(X_test_scaled)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy: %.2f%%" % (accuracy * 100.0))
'''
# 34.75 with normal svm classifier


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
'''param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    'kernel': ['rbf', 'poly', 'linear']  # Type of the kernel
}'''

param_grid = {
    'C': [5, 10, 15, 20],  # Narrowing down around the best C value
    'gamma': [0.005, 0.01, 0.015, 0.02],  # Narrowing down around the best gamma value
    'kernel': ['rbf']  # Since rbf seemed to work best
}


# Initialize the GridSearchCV object
# grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)


# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Print out the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with best model: %.2f%%" % (accuracy * 100.0))


from sklearn.ensemble import BaggingClassifier

bagging_model = BaggingClassifier(base_estimator=SVC(C=10, gamma=0.01, kernel='rbf'),
                                  n_estimators=10, random_state=42)

bagging_model.fit(X_train_scaled, y_train)
y_pred_bagging = bagging_model.predict(X_test_scaled)

accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print("Bagging with SVM Accuracy: %.2f%%" % (accuracy_bagging * 100.0))

