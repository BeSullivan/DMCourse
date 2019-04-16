import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 

data = pd.read_csv('../Dataset/Dataset2.csv')

Kcluster = 3
Eps = 0.25
minSamp = 4


kmeans = KMeans(n_clusters=Kcluster, random_state=0).fit(data)
kmeanLabels = kmeans.labels_

scaler = StandardScaler()
scaledData = scaler.fit_transform(data)
db = DBSCAN(eps= Eps, min_samples=minSamp, metric='euclidean')
dbClusters = db.fit_predict(scaledData)
dbLabels = db.labels_

dbscanSilScore = silhouette_score(data, dbLabels, metric='euclidean')
kmeanSilScore = silhouette_score(data, kmeanLabels, metric='euclidean')

plt.scatter(data['X'], data['Y'], c=kmeanLabels, cmap='plasma')
plt.title('Kmean - Silhoutte Score = {}'.format(kmeanSilScore))
plt.savefig('kmean_Task2_3_3.png')
plt.close()


plt.scatter(data['X'], data['Y'], c=dbLabels, cmap='plasma')
plt.title('DBSCAN - Silhoutte Score = {}'.format(dbscanSilScore))
plt.savefig('dbscan_Task2_3_3.png')
plt.close()

print ('Silhoutte Score for Kmean Method = {}'.format(kmeanSilScore))
print ('Silhoutte Score for DBSCAN Method = {}'.format(dbscanSilScore))











