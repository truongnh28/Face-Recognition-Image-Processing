import sys

import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog
from PyQt5.uic.properties import QtCore
from PyQt5 import QtCore

from ui import Ui_MainWindow
is_has_filtering = False
is_has_edge_detection = False
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.tmp = None
        self.kernel_size = (3,3)
        self.sigmaX = 0
        self.threshold1 = 100
        self.threshold2 = 200

        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        self.uic.open_button.clicked.connect(self.open_image)
        self.uic.filter_checkbox.stateChanged.connect(self.gaussian_blur)
        self.uic.edge_checkbox.stateChanged.connect(self.canny_edge_detection)
        self.uic.edge_detact_min_param.valueChanged.connect(self.canny_edge_param)
        self.uic.edge_detact_max_param.valueChanged.connect(self.canny_edge_param)
        # self.uic.pauseButton.clicked.connect(self.Pause)
        # self.uic.chooseFileButton.clicked.connect(self.ChooseFile)

    def show(self):
        self.main_win.show()

    def displayImage(self,image, window=1):
        qformat = QImage.Format_Indexed8

        if len(image.shape) == 3:
            if (image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img_temp = cv2.resize(image,(511,491))
        img = QImage(img_temp, img_temp.shape[1], img_temp.shape[0], img_temp.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped()  # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        # img = cv2.resize(img,(511,491))
        if window == 1:
            self.uic.image_before.setPixmap(QPixmap.fromImage(img))
            self.uic.image_before.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 2:
            self.uic.image_after.setPixmap(QPixmap.fromImage(img))
            self.uic.image_after.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    @pyqtSlot()
    def load_image(self, fname):
        self.image = cv2.imread(fname)
        # self.tmp = self.image
        self.displayImage(self.image)

    def open_image(self):
        image_file_name, filter = QFileDialog.getOpenFileName(self, 'Open File', r'~', "Image Files (*)")
        if image_file_name:
            self.load_image(image_file_name)
        else:
            print("Invalid Image")

    def gaussian_blur(self):
        global is_has_filtering
        is_has_filtering = not is_has_filtering
        image = cv2.GaussianBlur(self.image, self.kernel_size, self.sigmaX)
        if is_has_filtering:
            self.displayImage(image, window=2)
        else:
            self.displayImage(self.image, window=2)
    def canny_edge_detection(self):
        global is_has_edge_detection
        is_has_edge_detection = not is_has_edge_detection
        can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        can1 = cv2.GaussianBlur(can, self.kernel_size, self.sigmaX)
        image = cv2.Canny(can1, threshold1=self.uic.edge_detact_min_param.value(), threshold2=self.uic.edge_detact_max_param.value())
        if is_has_edge_detection:
            self.displayImage(image, window=2)
        else:
            self.displayImage(self.image,window=2)
    def canny_edge_param(self):
        can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        can1 = cv2.GaussianBlur(can, self.kernel_size, self.sigmaX)
        image = cv2.Canny(can1, threshold1=self.uic.edge_detact_min_param.value(),
                          threshold2=self.uic.edge_detact_max_param.value())
        if is_has_edge_detection:
            self.displayImage(image, window=2)
        else:
            self.displayImage(self.image, window=2)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
