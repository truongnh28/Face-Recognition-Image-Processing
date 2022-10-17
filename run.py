import sys

import cv2
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog
from PyQt5.uic.properties import QtCore
from PyQt5 import QtCore

from ui import Ui_MainWindow


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        self.uic.open_button.clicked.connect(self.open_image)
        # self.uic.pauseButton.clicked.connect(self.Pause)
        # self.uic.chooseFileButton.clicked.connect(self.ChooseFile)

    def show(self):
        self.main_win.show()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped()  # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        if window == 1:
            self.uic.image_before.setPixmap(QPixmap.fromImage(img))
            self.uic.image_before.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 2:
            pass
            # self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            # self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    @pyqtSlot()
    def load_image(self, fname):
        self.image = cv2.imread(fname)
        # self.tmp = self.image
        self.displayImage()

    def open_image(self):
        image_file_name, filter = QFileDialog.getOpenFileName(self, 'Open File', r'~', "Image Files (*)")
        if image_file_name:
            self.load_image(image_file_name)
        else:
            print("Invalid Image")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
