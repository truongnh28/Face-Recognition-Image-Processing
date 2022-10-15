import numpy as np
import cv2, sys
import pickle

labels = {}

with open("labels.pickle", 'rb') as f:
    ogLabels = pickle.load(f)
    labels = {v: k for k,v in ogLabels.items()}


# cap = cv2.VideoCapture(0)
# prefix = '/Users/lap01685/PycharmProjects/pythonProject/venv/lib/python3.8/site-packages/'
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '')
face_cascade = cv2.CascadeClassifier("/Users/lap01685/ProjectForImgProcessing/Face-Recognition-Image-Processing/cascades/data/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


def recognize(frame, x, y, roi_gray):
    id_, conf = recognizer.predict(roi_gray)
    print('conf: ', conf)
    if 10 <= conf <= 85:
        print(id_)
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = [255, 255, 255]
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


def detect_and_recog(prefix, photo_path):
        #ret, frame = cap.read()
    frame = cv2.imread(photo_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print(x, y, w, h, '\n')
        # print("here", gray, '\n')
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # print('roi_gay', roi_gray, '\n')

        img_item = prefix + "images/my-image3.png"

        color = (255, 0, 0)
        stroke = 2
        x_end_cord = x + w
        y_end_cord = y + h
        cv2.rectangle(frame, (x, y), (x_end_cord, y_end_cord), color, stroke)
        print("here")
        # recognizer
        recognize(frame, x, y, roi_gray)
        cv2.imwrite(img_item, frame)

    cv2.imshow('frame', frame)


    # cap.release()
    cv2.destroyAllWindows()


def main():
    detect_and_recog('', "/Users/lap01685/ProjectForImgProcessing/Face-Recognition-Image-Processing/src/images/mypic01.jpg")


if __name__ == "__main__":
    main()





