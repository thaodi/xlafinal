import cv2
import numpy as np
import os

def label_to_integer(label):
    if label.isalpha():
        return ord(label.lower()) - ord('a') + 10
    return int(label)

imgs = []
labels = []

data_path = 'dataset/'

for fd in os.listdir(data_path):
    fd_path = data_path + fd + '/'
    for f in os.listdir(fd_path):
        img_path = os.path.join(fd_path, f)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 128))
        features = cv2.HOGDescriptor().compute(img)
        if not img is None:
            imgs.append(features)
            labels.append(label_to_integer(fd))

imgs = np.array(imgs, dtype="float32")
labels = np.array(labels, dtype="int32")

knn = cv2.ml.KNearest_create()
knn.train(imgs, cv2.ml.ROW_SAMPLE, labels)

img = cv2.imread('test.jpg')
img_t = img.copy()
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(image_gray, 255, 1, 1, 11, 2)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

# Vẽ khung chữ nhật xung quanh mỗi contour
for contour in contours:
    if cv2.contourArea(contour) > 120:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20 and h < 30:
            nb = img_t[y:y+h, x:x+w]
            nb = cv2.resize(nb, (64, 128))
            nb = cv2.HOGDescriptor().compute(nb)
            nb = np.array(nb, dtype="float32")
            nb = nb.reshape(1, nb.shape[0])
            retval, results, neigh_resp, dists = knn.findNearest(nb, k = 1)
            if int(results[0][0]) < 10:
                cv2.rectangle(img_t, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_text = str(int(results[0][0]))
                cv2.putText(img_t, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow('result.jpg', img_t)
cv2.waitKey(0)