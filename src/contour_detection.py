import numpy as np
import cv2

image_path = "/app/samples/bagtag_0.png"
image = cv2.imread(image_path)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

image_path_contour = "/app/samples/bagtag_0_contour.jpg"
cv2.imwrite(image_path_contour, image_copy)