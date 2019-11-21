# https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


img = cv2.imread('out/bg_removed.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

pixels_list = img.reshape((img.shape[0] * img.shape[1], 3))

clt = KMeans(n_clusters=7, n_init=40, max_iter=500)
kmeans = clt.fit(pixels_list)

print(clt.labels_)
print(np.arange(0, len(np.unique(clt.labels_)) + 1))
print(clt.cluster_centers_)
exit()

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

#
# plt.figure()
# plt.axis("off")
# plt.subplot(121), plt.imshow(img)
# plt.subplot(122), plt.imshow(bar)
# plt.show()
