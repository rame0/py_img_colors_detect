import cv2
import numpy as np
import matplotlib.pyplot as plt
import _utils
from mpl_toolkits.mplot3d import Axes3D

# Load Pallet
from palets.minkukel_com import pallet_color
from palets.minkukel_com import pallet_color_name as cluster_names

clusters = np.array(pallet_color, np.uint8)

img = cv2.imread('out/bg_removed.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
# convert image to pixels list
pixels_list = np.reshape(img, (img.shape[0] * img.shape[1], 3))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], c=clusters / 255)
plt.show()

distances = []
pixel_labels = np.zeros(len(pixels_list), dtype=np.int8)
pixel_img_clust = np.full(pixels_list.shape, (0, 0, 255), dtype=np.uint8)

for p_key, point in enumerate(pixels_list):
    min_distance = 50000000  # max possible distance in RGB is about 441
    # skip removed BG
    if point[0] < 5 and point[1] < 5 and point[2] > 250:  # RGB
        pixel_labels[p_key] = -1
        continue

    for c_key, cluster_color in enumerate(clusters):
        distance = _utils.distance_sq(point, cluster_color)
        # distance = cv2.norm(point - cluster_color, cv2.NORM_L2)
        if min_distance > distance:
            pixel_labels[p_key] = c_key
            pixel_img_clust[p_key] = cluster_color
            min_distance = distance

numLabels = np.arange(0, len(cluster_names) + 1)
(hist, _) = np.histogram(pixel_labels, bins=numLabels)
hist = hist.astype("float")
hist /= hist.sum()

bar_data = np.array(list(zip(cluster_names, clusters, hist)))
# bar_data = bar_data[bar_data[:, 2] > .02]
# print(hist)
# print(bar_data)

bar = _utils.plot_colors(bar_data)

# reshape image only with cluster colors
# for testing
# pixel_img_clust = np.reshape(pixel_img_clust, (img.shape[0], img.shape[1], 3))


pixel_img_clust = pixel_img_clust.reshape((img.shape))

plt.axis("off")
# RGB
plt.subplot2grid((2, 2), (0, 0)), plt.imshow(img)
plt.subplot2grid((2, 2), (0, 1)), plt.imshow(pixel_img_clust)
plt.subplot2grid((2, 2), (1, 0), colspan=2), plt.imshow(bar)
plt.show()
