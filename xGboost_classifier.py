import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

raw_df= pd.read_csv("../data-final.csv", sep='\t')
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

top_countries = dataset['country'].value_counts().nlargest(4).index
filtered_dataset = dataset[dataset['country'].isin(top_countries)]
plt.figure(figsize=(8, 6))
sns.countplot(data=filtered_dataset, x='country', palette='viridis')
plt.xlabel('Country', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Top 4 Countries by Participation', fontweight='bold')
plt.show()

'''
dataset['nation'] = dataset['country'].apply(lambda x: 1 if x =='US' else (2 if x =='GB' else (3 if x=='CA' else (4 if x=='AU' else 0))))
dataset.drop("country", axis=1, inplace=True)

dataset = dataset[dataset['nation'] != 0]
dataset['nation'] = dataset['nation'].map({1: 0, 2: 1, 3: 2, 4: 3})

plt.figure(figsize=(8, 6))
sns.countplot(data=dataset, x='nation', palette='viridis')
plt.xlabel('Nation', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Distribution of Responses by Nation', fontweight='bold')
#plt.show()

dataset = dataset.sample(n=len(dataset),random_state=42)
y = dataset['nation']
X = dataset.drop('nation',axis=1)

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X,y)
print('Original data size:\n',y.value_counts())
print('Data size after SMOTE is done:\n',y_smote.value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x=y_smote, palette='viridis')
plt.xlabel('Nation', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Distribution of Responses by Nation', fontweight='bold')
#plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.1, random_state=42)

xgb_model = xgb.XGBClassifier(learning_rate=1,
                              max_depth=6,
                              gamma=0.08435594187707007,
                              colsample_bytree=0.5336629698328548,
                              n_estimators=10000,
                              objective='binary:logistic',
                              random_state=42,
                              early_stopping_rounds=10)

xgb_model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test,y_test)])

y_pred = xgb_model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Accuracy: 74%
'''
'''
xgb_model = xgb.XGBClassifier(learning_rate=1,
                              max_depth=10,
                              gamma=0.08435594187707007,
                              colsample_bytree=0.5336629698328548,
                              n_estimators=1000,
                              objective='binary:logistic',
                              random_state=42,
                              early_stopping_rounds=10)

xgb_model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test,y_test)])

y_pred = xgb_model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''
# Accuracy: 72%