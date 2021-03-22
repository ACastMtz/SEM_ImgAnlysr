import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.filters
import skimage.measure
import sklearn.metrics
import cv2 
import itertools

img_path = 'Images/NW_test01.jpg'
img = cv2.imread(img_path)
grayscale = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

# KMeans thresholding

grayscale_n = grayscale.reshape(grayscale.shape[0]*grayscale.shape[1],1)/255

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(grayscale_n)
img2show = kmeans.cluster_centers_[kmeans.labels_]
print(np.sort(kmeans.cluster_centers_, axis=0))
cluster_img = img2show.reshape(
    grayscale.shape[0],
    grayscale.shape[1]
)
threshold = np.sort(kmeans.cluster_centers_, axis=0)

# Find connected components

# Binarize image
thr = int(threshold[1][0]*255)
ret, bw = cv2.threshold(grayscale, thr, 255, cv2.THRESH_BINARY)
# bw = grayscale > threshold[2]

# Find connected components
connectivity = 8
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
sizes = stats[1:, -1] 
nb_components = nb_components - 1
min_size = 250 #threshhold value for objects in scene
img2 = np.zeros((img.shape), np.uint8)
for i in range(0, nb_components+1):
    color = np.random.randint(255,size=3)
    cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
    img2[output == i + 1] = color

print('Threshold: {}'.format(thr))
print('Number of labels: {}'.format(nb_components))
print('Label matrix: {}'.format(output))

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.imshow(img)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img2)
plt.show()
