import numpy as np 
import pandas as pd 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('../Dataset/Dataset2.csv')

scaler = StandardScaler()
scaledData = scaler.fit_transform(data)

Eps = 1
min_sample = 1
db = DBSCAN(eps=Eps ,min_samples=min_sample, metric='euclidean')
clusters = db.fit_predict(scaledData)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
num_noise = list(labels).count(-1)

plt.scatter(data['X'], data['Y'] ,c=clusters ,cmap='plasma')
plt.title('DBSCAN with eps = {} and min_sample = {}'.format(Eps, min_sample))
plt.savefig('dbscan_0_cluster.png')
plt.close()

