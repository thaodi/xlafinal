import cv2
import numpy as np
input_image = cv2.imread("test4.jpg")
kernel = np.ones((5,5),np.uint8)


hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
lower = (0, 0, 0)  # Lower HSV values for star color
upper = (179, 255,95)   # Upper HSV values for star color

mask = cv2.inRange(hsv_image, lower,upper)
ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("14.jpg",thresh)
cv2.waitKey(0) 
cv2.destroyAllWindows() 