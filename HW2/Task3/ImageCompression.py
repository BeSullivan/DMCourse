import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from matplotlib import image, pyplot

img = image.imread('../Dataset/imageLarge.png')

rows = img.shape[0]
cols = img.shape[1]

imgReshaped = (img / 255.0).reshape(img.shape[0] * img.shape[1], 3)

numColor = 256
kmean = KMeans(n_clusters=numColor, max_iter=50).fit(imgReshaped)

img16 = kmean.cluster_centers_[kmean.labels_]
img16 = img16 * 255.0
img16 = np.reshape(img16 ,(rows ,cols ,3))

image.imsave('img256.png', img16)

f = pyplot.figure()
f.add_subplot(1, 2, 1)
pyplot.title('Main Picture')
pyplot.imshow(img)
f.add_subplot(1, 2, 2)
pyplot.title('Compressed Picture 256 colors')
pyplot.imshow(img16)
pyplot.show(block=True)
f.savefig('Comparing256.png')
pyplot.close()