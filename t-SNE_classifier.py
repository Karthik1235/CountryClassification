import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming 'X' is your dataset without labels

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can change the number of clusters
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster information to the dataset
X_clustered = X.copy()
X_clustered['Cluster'] = clusters

pca = PCA(n_components=2)  # Reducing data to two dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# Assuming 'X_scaled' is your scaled dataset without labels
# and 'clusters' are the cluster assignments from KMeans

# Initialize t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

# Fit and transform the data
X_tsne = tsne.fit_transform(X_scaled)

# Plot the t-SNE transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', marker='o')
plt.colorbar(label='Cluster')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.show()
