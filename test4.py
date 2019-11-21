import cv2
import matplotlib.pyplot as plt

dim = (968, 727)

bg_image = cv2.imread('img/test3.jpg')
bg_image = cv2.resize(bg_image, dim, interpolation=cv2.INTER_AREA)

bg_im_gray_blur = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
# bg_im_gray_blur = cv2.GaussianBlur(bg_im_gray_blur, (5, 5), 0)

image = cv2.imread('img/test2.jpg')
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

im_gray_blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# im_gray_blur = cv2.GaussianBlur(im_gray_blur, (5, 5), 0)

difference = cv2.absdiff(bg_im_gray_blur, im_gray_blur)
_, difference = cv2.threshold(difference, 5, 255, cv2.THRESH_BINARY)

plt.imshow(im_gray_blur), plt.show()
plt.imshow(bg_image), plt.show()
plt.imshow(difference), plt.show()
