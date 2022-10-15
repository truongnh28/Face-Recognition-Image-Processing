import os
from PIL import Image
import numpy as np
import cv2
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("base: ", BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")
face_cascade = cv2.CascadeClassifier("/Users/lap01685/ProjectForImgProcessing/Face-Recognition-Image-Processing/cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            # y_label.append(label)
            # x_train.append(path)
            pil_image = Image.open(path).convert("L")
            # size = [550, 550]
            # final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            # print('this is id1: ', id_)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                # print("this is id: ", id_)
                y_labels.append(id_)

# print("this")
# print(x_train)
# print("after this line")
# print(y_labels)


with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
