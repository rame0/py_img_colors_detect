import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("img_test/11.jpg")
Z = img.reshape((-1, 3))

# calc clusters and decolorize image
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
decolorised = res.reshape(img.shape)

# creating mask
edges = cv2.Canny(decolorised, 50, 150)
# applying closing function
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# apply mask
masked = cv2.bitwise_or(img, img, mask=mask)

# plot
plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(cv2.cvtColor(decolorised, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Decolorised'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(mask, cmap='gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Masked'), plt.xticks([]), plt.yticks([])

plt.show()
