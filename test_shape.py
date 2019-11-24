import cv2
import numpy as np
from matplotlib import pyplot as plt


def process(img):
    work_img = img.copy()

    # _, threshold = cv2.threshold(work_img, 137, 255, 0)
    _, threshold = cv2.threshold(work_img, 30, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

        cnt_area = cv2.contourArea(cnt)
        if len(approx) > 3 and cnt_area > max_area:
            max_area = cnt_area
            max_contour = approx

    cv2.drawContours(work_img, [max_contour], 0, (255, 0, 0), 2)

    return work_img, threshold


img = cv2.imread("img_test/1.jpg", 0)
img_blur = cv2.blur(img, (20, 20))

res_img, threshold = process(img)

plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)), plt.title('Image')
plt.subplot(232), plt.imshow(threshold), plt.title('Threshold')
plt.subplot(233), plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_GRAY2RGB)), plt.title('Largest shape')

res_img, threshold = process(img_blur)
plt.subplot(234), plt.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)), plt.title('Image blurred')
plt.subplot(235), plt.imshow(threshold), plt.title('Threshold from blurred')
plt.subplot(236), plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_GRAY2RGB)), plt.title('Largest shape')

plt.show()
