import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

raw_df= pd.read_csv("../data-final.csv", sep='\t', nrows = 2000)
dataset=raw_df.copy()

# Drop unwanted columns
dataset.drop(dataset.columns[50:107], axis=1, inplace=True)
dataset.drop(dataset.columns[51:], axis=1, inplace=True)

# Drop missing values
print('Is there any missing value? ', dataset.isnull().values.any())
print('How many missing values? ', dataset.isnull().values.sum())
dataset.dropna(inplace=True)
print('Number of participants after eliminating missing values: ', len(dataset))

# Classify into positive and negative columns/ attributes
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

# For positive, replace lower rating with negative values
# For negative, replace higher values with higher positive values
dataset[positive] = dataset[positive].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
dataset[negative] = dataset[negative].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})
cols = positive + negative
database = dataset[sorted(cols) + ['country']]

EXT = list(database.columns[:10])
EST = list(database.columns[10:20])
AGR = list(database.columns[20:30])
CSN = list(database.columns[30:40])
OPN = list(database.columns[40:50])

# Dimensions based on which prediction is performed
dimensions = [EXT,EST,AGR,CSN,OPN]
dimension_averages=["extraversion","neuroticism","agreeableness","conscientiousness","openness"]

# take mean value for each of the dimension's attributes
for d in range(len(dimensions)):
    dataset[dimension_averages[d]] = dataset[dimensions[d]].mean(axis=1)

# Since there are 4 nations whose data exceeds compared to other nations
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



X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.1, random_state=42)

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Only transform the test data
X_test_scaled = scaler.transform(X_test)



from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
min_max_scaler = MinMaxScaler()

# Fit and transform the training data
X_train_normalized = min_max_scaler.fit_transform(X_train)

# Only transform the test data
X_test_normalized = min_max_scaler.transform(X_test)



# Initialize the Gaussian Naive Bayes classifier
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Making predictions
y_pred = nb_model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
