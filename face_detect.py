import cv2
import pathlib

'''
Hàm face_detect sẽ trả ra một ma trận ảnh đã có xác định khuôn mặt
Tham số:
  - img: ma trận ảnh ban đầu
  - scale_factor: nếu không truyền sẽ mặc định là 1.01, ổn định nhất là 1.05
Cách dùng:
  img = cv2.imread('/data/img.png') # Đọc ảnh từ đường dẫn ra một biến
  new_img = face_detect(img, 1.05) # Detect khuôn mặt
  cv2.imwrite('/data/img-detect.png', new_img) # Ghi ma trận nhận được ra một file ảnh mới
'''


def face_detect(img, scale_factor=1.01):
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


if __name__ == '__main__':
    img = cv2.imread("./resource/check.png")
    face_detect(img, 1.04)
    print("cc")
