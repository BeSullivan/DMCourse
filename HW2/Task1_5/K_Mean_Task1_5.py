import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math 

data = pd.read_csv('../Dataset/Dataset2.csv')

n = 15
dataSize = len(data)

def initialCenters(data ,K):
    rnd = np.random.random_integers(0 ,dataSize - 1 ,K)
    centers = []
    for i in range(0 ,K):
        centers.append([data.iloc[rnd[i]]['X'], data.iloc[rnd[i]]['Y']])
    return centers

def distEuclidean(c1 ,c2):
    return pow((c1[0] - c2['X']) , 2) + pow((c1[1] - c2['Y']) , 2)

def findClosestCentroids(data, centroids):
    idx = []
    for i in range(0 ,dataSize):
        idx.append(0)
        dist = 100
        for j in range(0 ,K):
            tmp = distEuclidean(centroids[j] ,data.iloc[i])
            if dist > tmp:
                dist = tmp
                idx[i] = j
    return idx

def computeCenters(data, idx, K):
    newCenters = []
    num = []
    for i in range(0 ,K):
        newCenters.append([0 ,0])
        num.append(0)

    for i in range(0, dataSize):
        newCenters[idx[i]][0] += data.iloc[i][0]
        newCenters[idx[i]][1] += data.iloc[i][1]
        num[idx[i]] += 1

    for i in range(0 ,K):
        newCenters[i][0] /= float(num[i])
        newCenters[i][1] /= float(num[i])

    return newCenters

def calcError(data, idx, centroids, K):
    sumErr = 0.0
    err = []
    num = []

    for i in range(0 ,K):
        err.append(0)  
        num.append(0)

    for i in range(0, dataSize):
        err[idx[i]] += math.sqrt( distEuclidean(centroids[idx[i]], data.iloc[i]) )
        num[idx[i]] += 1

    for i in range(0, K): 
        err[i] /= float(num[i])
        sumErr += err[i]

    return sumErr

def Scatter(data, idx, centroids ,K):

    plt.scatter(data['X'], data['Y'], c=idx, s=10, cmap='viridis')
    xCenter = []
    yCenter = []
    for i in range(0, K):
        xCenter.append(centroids[i][0])
        yCenter.append(centroids[i][1])
    plt.scatter(xCenter, yCenter, c='black' ,marker='x')
    plt.savefig('Task1_5_Clusters_#{}.png'.format(K))
    plt.close()

xAllErr = []
yAllErr =[]

for K in range(2 ,6):

    print('Number of Clusters : {}'.format(K),end='\n')

    centroids = initialCenters(data ,K)

    for i in range(0, n):
        idx = findClosestCentroids(data, centroids)
        centroids = computeCenters(data ,idx ,K)

    Scatter(data ,idx ,centroids ,K)

    xAllErr.append(K)
    yAllErr.append(calcError(data, idx, centroids, K) / K)

plt.plot(xAllErr, yAllErr ,c='black' ,linestyle='--', marker='o')
plt.savefig('Task1_5_Error.png')
plt.show()


