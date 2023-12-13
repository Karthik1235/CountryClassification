import pandas as pd
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

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.1, random_state=42)

'''from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: %.2f%%" % (accuracy_rf * 100.0))'''


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, 30],        # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required at each leaf node
}

# Initialize the GridSearchCV object
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Use the best model to make predictions
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Calculate and print the accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Improved Random Forest Accuracy: %.2f%%" % (accuracy_rf * 100.0))

'''
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# Get feature importances from the Random Forest model
importances = best_rf_model.feature_importances_

# Convert the importances into a DataFrame
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Plot the feature importances
plt.figure(figsize=(12, 6))
feature_importances.sort_values(by='importance', ascending=False).head(10).set_index('feature').plot(kind='bar')
plt.title('Top 10 Most Important Features')
plt.show()


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: ", cv_scores.mean())
'''