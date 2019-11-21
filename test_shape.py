import cv2
import numpy as np
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

img = cv2.imread("img/202.jpg", 0)
img_blur = cv2.blur(img, (20, 20))

# threshold = cv2.Canny(img_blur, 30, 30)
ret, threshold = cv2.threshold(img, 137, 255, 0)
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
max_contour = [];
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 3:
        # cv2.putText(img, "Triangle", (x, y), font, 1, 0)
        continue
    elif len(approx) == 4:
        print(approx)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > max_area:
            max_area = cnt_area
            max_contour = approx
        # cv2.putText(img, "Rectangle", (x, y), font, 1, 0)
    elif len(approx) == 5:
        # cv2.putText(img, "Pentagon", (x, y), font, 1, 0)
        continue
    elif 6 < len(approx) < 15:
        # cv2.putText(img, "Ellipse", (x, y), font, 1, 0)
        continue
    else:
        # cv2.putText(img, "Circle", (x, y), font, 1, 0)
        continue

    # cv2.drawContours(img, [approx], 0, 0, 2)

print(max_area)
cv2.drawContours(img, [max_contour], 0, 0, 2)

plt.subplot(131), plt.imshow(img), plt.title('Shapes')
plt.subplot(132), plt.imshow(img_blur), plt.title('blurred')
plt.subplot(133), plt.imshow(threshold), plt.title('Threshold')
plt.show()
