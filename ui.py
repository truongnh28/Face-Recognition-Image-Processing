# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1440, 876)
        MainWindow.setStyleSheet("background-color:rgb(255,255,255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.properties_frame = QtWidgets.QFrame(self.centralwidget)
        self.properties_frame.setGeometry(QtCore.QRect(1071, 70, 351, 691))
        self.properties_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.properties_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.properties_frame.setObjectName("properties_frame")
        self.reset_button = QtWidgets.QPushButton(self.properties_frame)
        self.reset_button.setGeometry(QtCore.QRect(40, 590, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.reset_button.setFont(font)
        self.reset_button.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.reset_button.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.reset_button.setAutoFillBackground(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../Downloads/reset.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.reset_button.setIcon(icon)
        self.reset_button.setIconSize(QtCore.QSize(25, 25))
        self.reset_button.setAutoRepeat(False)
        self.reset_button.setAutoExclusive(False)
        self.reset_button.setAutoDefault(False)
        self.reset_button.setDefault(True)
        self.reset_button.setFlat(True)
        self.reset_button.setObjectName("reset_button")
        self.ok_button = QtWidgets.QPushButton(self.properties_frame)
        self.ok_button.setGeometry(QtCore.QRect(200, 590, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.ok_button.setFont(font)
        self.ok_button.setFocusPolicy(QtCore.Qt.StrongFocus)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../Downloads/check.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ok_button.setIcon(icon1)
        self.ok_button.setIconSize(QtCore.QSize(25, 25))
        self.ok_button.setDefault(True)
        self.ok_button.setFlat(True)
        self.ok_button.setObjectName("ok_button")
        self.edge_detection_frame = QtWidgets.QFrame(self.properties_frame)
        self.edge_detection_frame.setGeometry(QtCore.QRect(10, 250, 331, 121))
        self.edge_detection_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.edge_detection_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.edge_detection_frame.setObjectName("edge_detection_frame")
        self.label_14 = QtWidgets.QLabel(self.edge_detection_frame)
        self.label_14.setGeometry(QtCore.QRect(10, 10, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.edge_detection_frame)
        self.label_15.setGeometry(QtCore.QRect(10, 50, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.edge_detection_frame)
        self.label_16.setGeometry(QtCore.QRect(10, 80, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.edge_detact_min_param = QtWidgets.QSlider(self.edge_detection_frame)
        self.edge_detact_min_param.setGeometry(QtCore.QRect(120, 40, 201, 31))
        # self.edge_detact_min_param.setMinimum(1)
        self.edge_detact_min_param.setMaximum(200)
        self.edge_detact_min_param.setProperty("value", 75)
        # self.edge_detact_min_param.setSliderPosition(1)
        self.edge_detact_min_param.setOrientation(QtCore.Qt.Horizontal)
        self.edge_detact_min_param.setObjectName("edge_detact_min_param")
        self.edge_detact_max_param = QtWidgets.QSlider(self.edge_detection_frame)
        self.edge_detact_max_param.setGeometry(QtCore.QRect(120, 70, 201, 31))
        self.edge_detact_max_param.setMaximum(300)
        self.edge_detact_max_param.setProperty("value", 150)
        self.edge_detact_max_param.setOrientation(QtCore.Qt.Horizontal)
        self.edge_detact_max_param.setObjectName("edge_detact_max_param")
        self.face_detection_frame = QtWidgets.QFrame(self.properties_frame)
        self.face_detection_frame.setGeometry(QtCore.QRect(10, 390, 331, 131))
        self.face_detection_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.face_detection_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.face_detection_frame.setObjectName("face_detection_frame")
        self.label_17 = QtWidgets.QLabel(self.face_detection_frame)
        self.label_17.setGeometry(QtCore.QRect(10, 10, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.face_detection_frame)
        self.label_18.setGeometry(QtCore.QRect(10, 70, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.scale_factor_param = QtWidgets.QSlider(self.face_detection_frame)
        self.scale_factor_param.setGeometry(QtCore.QRect(120, 51, 201, 31))
        self.scale_factor_param.setMinimum(101)
        self.scale_factor_param.setMaximum(150)
        self.scale_factor_param.setProperty("value", 105)
        self.scale_factor_param.setSliderPosition(1)
        self.scale_factor_param.setOrientation(QtCore.Qt.Horizontal)
        self.scale_factor_param.setObjectName("scale_factor_param")
        self.label_10 = QtWidgets.QLabel(self.properties_frame)
        self.label_10.setGeometry(QtCore.QRect(120, 10, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.filtering_frame = QtWidgets.QFrame(self.properties_frame)
        self.filtering_frame.setGeometry(QtCore.QRect(10, 50, 331, 181))
        self.filtering_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.filtering_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.filtering_frame.setObjectName("filtering_frame")
        self.label_7 = QtWidgets.QLabel(self.filtering_frame)
        self.label_7.setGeometry(QtCore.QRect(10, 149, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.filtering_frame)
        self.label_8.setGeometry(QtCore.QRect(10, 119, 55, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_11 = QtWidgets.QLabel(self.filtering_frame)
        self.label_11.setGeometry(QtCore.QRect(10, 89, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.filtering_frame)
        self.label_12.setGeometry(QtCore.QRect(10, 59, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.bilateral_param = QtWidgets.QSlider(self.filtering_frame)
        self.bilateral_param.setGeometry(QtCore.QRect(120, 140, 201, 31))
        self.bilateral_param.setMinimum(1)
        self.bilateral_param.setMaximum(10)
        self.bilateral_param.setPageStep(1)
        self.bilateral_param.setProperty("value", 9)
        self.bilateral_param.setSliderPosition(9)
        self.bilateral_param.setOrientation(QtCore.Qt.Horizontal)
        self.bilateral_param.setObjectName("bilateral_param")
        self.gaussian_param = QtWidgets.QSlider(self.filtering_frame)
        self.gaussian_param.setGeometry(QtCore.QRect(120, 50, 201, 31))
        self.gaussian_param.setMinimum(0)
        self.gaussian_param.setMaximum(4)
        self.gaussian_param.setPageStep(10)
        self.gaussian_param.setProperty("value", 0)
        self.gaussian_param.setSliderPosition(0)
        self.gaussian_param.setOrientation(QtCore.Qt.Horizontal)
        self.gaussian_param.setObjectName("gaussian_param")
        self.normalized_param = QtWidgets.QSlider(self.filtering_frame)
        self.normalized_param.setGeometry(QtCore.QRect(120, 80, 201, 31))
        self.normalized_param.setMinimum(0)
        self.normalized_param.setMaximum(3)
        self.normalized_param.setPageStep(10)
        self.normalized_param.setProperty("value", 0)
        self.normalized_param.setSliderPosition(0)
        self.normalized_param.setOrientation(QtCore.Qt.Horizontal)
        self.normalized_param.setObjectName("normalized_param")
        self.median_param = QtWidgets.QSlider(self.filtering_frame)
        self.median_param.setGeometry(QtCore.QRect(120, 110, 201, 31))
        self.median_param.setMinimum(0)
        self.median_param.setMaximum(3)
        self.median_param.setPageStep(1)
        self.median_param.setProperty("value", 0)
        self.median_param.setSliderPosition(0)
        self.median_param.setOrientation(QtCore.Qt.Horizontal)
        self.median_param.setObjectName("median_param")
        self.label_13 = QtWidgets.QLabel(self.filtering_frame)
        self.label_13.setGeometry(QtCore.QRect(10, 10, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.main_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_frame.setGeometry(QtCore.QRect(20, 70, 1041, 691))
        self.main_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_frame.setObjectName("main_frame")
        self.image_before = QtWidgets.QLabel(self.main_frame)
        self.image_before.setGeometry(QtCore.QRect(10, 50, 500, 490))
        self.image_before.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.image_before.setFrameShape(QtWidgets.QFrame.Panel)
        self.image_before.setFrameShadow(QtWidgets.QFrame.Plain)
        self.image_before.setText("")
        self.image_before.setObjectName("image_before")
        self.image_after = QtWidgets.QLabel(self.main_frame)
        self.image_after.setGeometry(QtCore.QRect(530, 50, 500, 490))
        self.image_after.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.image_after.setFrameShape(QtWidgets.QFrame.Panel)
        self.image_after.setFrameShadow(QtWidgets.QFrame.Plain)
        self.image_after.setText("")
        self.image_after.setObjectName("image_after")
        self.filter_checkbox = QtWidgets.QCheckBox(self.main_frame)
        self.filter_checkbox.setGeometry(QtCore.QRect(150, 610, 101, 20))
        self.filter_checkbox.setObjectName("filter_checkbox")
        self.edge_checkbox = QtWidgets.QCheckBox(self.main_frame)
        self.edge_checkbox.setGeometry(QtCore.QRect(370, 610, 121, 20))
        self.edge_checkbox.setObjectName("edge_checkbox")
        self.face_detect_checkbox = QtWidgets.QCheckBox(self.main_frame)
        self.face_detect_checkbox.setGeometry(QtCore.QRect(580, 610, 121, 20))
        self.face_detect_checkbox.setObjectName("face_detect_checkbox")
        self.face_recog_checkbox = QtWidgets.QCheckBox(self.main_frame)
        self.face_recog_checkbox.setGeometry(QtCore.QRect(810, 610, 131, 20))
        self.face_recog_checkbox.setObjectName("face_recog_checkbox")
        self.before_label = QtWidgets.QLabel(self.main_frame)
        self.before_label.setGeometry(QtCore.QRect(190, 20, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.before_label.setFont(font)
        self.before_label.setObjectName("before_label")
        self.after_label = QtWidgets.QLabel(self.main_frame)
        self.after_label.setGeometry(QtCore.QRect(740, 20, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.after_label.setFont(font)
        self.after_label.setObjectName("after_label")
        self.tool_frame = QtWidgets.QFrame(self.centralwidget)
        self.tool_frame.setGeometry(QtCore.QRect(20, 10, 1401, 51))
        self.tool_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tool_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tool_frame.setObjectName("tool_frame")
        self.open_button = QtWidgets.QPushButton(self.tool_frame)
        self.open_button.setGeometry(QtCore.QRect(0, 0, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.open_button.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../../Downloads/open-folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.open_button.setIcon(icon2)
        self.open_button.setIconSize(QtCore.QSize(25, 25))
        self.open_button.setDefault(True)
        self.open_button.setFlat(True)
        self.open_button.setObjectName("open_button")
        self.save_button = QtWidgets.QPushButton(self.tool_frame)
        self.save_button.setGeometry(QtCore.QRect(120, 0, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.save_button.setFont(font)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../../Downloads/floppy-disk.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_button.setIcon(icon3)
        self.save_button.setIconSize(QtCore.QSize(25, 25))
        self.save_button.setDefault(True)
        self.save_button.setFlat(True)
        self.save_button.setObjectName("save_button")
        self.export_button = QtWidgets.QPushButton(self.tool_frame)
        self.export_button.setGeometry(QtCore.QRect(240, 0, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.export_button.setFont(font)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../../Downloads/printer.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.export_button.setIcon(icon4)
        self.export_button.setIconSize(QtCore.QSize(25, 25))
        self.export_button.setDefault(True)
        self.export_button.setFlat(True)
        self.export_button.setObjectName("export_button")
        self.exit_button = QtWidgets.QPushButton(self.tool_frame)
        self.exit_button.setGeometry(QtCore.QRect(360, 0, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.exit_button.setFont(font)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../../Downloads/logout.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.exit_button.setIcon(icon5)
        self.exit_button.setIconSize(QtCore.QSize(25, 25))
        self.exit_button.setDefault(True)
        self.exit_button.setFlat(True)
        self.exit_button.setObjectName("exit_button")
        self.main_frame.raise_()
        self.properties_frame.raise_()
        self.tool_frame.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.image = None
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionOpen.setIcon(icon2)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setIcon(icon3)
        self.actionSave.setObjectName("actionSave")
        self.actionPrint = QtWidgets.QAction(MainWindow)
        self.actionPrint.setIcon(icon4)
        self.actionPrint.setObjectName("actionPrint")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setIcon(icon5)
        self.actionQuit.setObjectName("actionQuit")
        self.actioAnhXam = QtWidgets.QAction(MainWindow)
        self.actioAnhXam.setObjectName("actioAnhXam")
        self.actionCanny = QtWidgets.QAction(MainWindow)
        self.actionCanny.setObjectName("actionCanny")
        self.actionNegative = QtWidgets.QAction(MainWindow)
        self.actionNegative.setObjectName("actionNegative")
        self.actionHistogram = QtWidgets.QAction(MainWindow)
        self.actionHistogram.setObjectName("actionHistogram")
        self.actionLog = QtWidgets.QAction(MainWindow)
        self.actionLog.setObjectName("actionLog")
        self.actionGaussan = QtWidgets.QAction(MainWindow)
        self.actionGaussan.setObjectName("actionGaussan")
        self.actionHigh_Boost = QtWidgets.QAction(MainWindow)
        self.actionHigh_Boost.setObjectName("actionHigh_Boost")
        self.actionLaplacian = QtWidgets.QAction(MainWindow)
        self.actionLaplacian.setObjectName("actionLaplacian")
        self.actionFilter_Average = QtWidgets.QAction(MainWindow)
        self.actionFilter_Average.setObjectName("actionFilter_Average")
        self.actionUnsharp = QtWidgets.QAction(MainWindow)
        self.actionUnsharp.setObjectName("actionUnsharp")
        self.actionCh_ng_5 = QtWidgets.QAction(MainWindow)
        self.actionCh_ng_5.setObjectName("actionCh_ng_5")
        self.actionTanSo = QtWidgets.QAction(MainWindow)
        self.actionTanSo.setObjectName("actionTanSo")
        self.actionIdeal_LPF = QtWidgets.QAction(MainWindow)
        self.actionIdeal_LPF.setObjectName("actionIdeal_LPF")
        self.actionButter_LPF = QtWidgets.QAction(MainWindow)
        self.actionButter_LPF.setObjectName("actionButter_LPF")
        self.actionGaussian_LPF = QtWidgets.QAction(MainWindow)
        self.actionGaussian_LPF.setObjectName("actionGaussian_LPF")
        self.actionIdeal_HPF = QtWidgets.QAction(MainWindow)
        self.actionIdeal_HPF.setObjectName("actionIdeal_HPF")
        self.actionButterworth_HPF = QtWidgets.QAction(MainWindow)
        self.actionButterworth_HPF.setObjectName("actionButterworth_HPF")
        self.actionGaussian_HPF = QtWidgets.QAction(MainWindow)
        self.actionGaussian_HPF.setObjectName("actionGaussian_HPF")
        self.actiondilate = QtWidgets.QAction(MainWindow)
        self.actiondilate.setObjectName("actiondilate")
        self.actionErode = QtWidgets.QAction(MainWindow)
        self.actionErode.setObjectName("actionErode")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionHit_miss = QtWidgets.QAction(MainWindow)
        self.actionHit_miss.setObjectName("actionHit_miss")
        self.actionDilate = QtWidgets.QAction(MainWindow)
        self.actionDilate.setObjectName("actionDilate")
        self.actionMorboundary = QtWidgets.QAction(MainWindow)
        self.actionMorboundary.setObjectName("actionMorboundary")
        self.actionGradient = QtWidgets.QAction(MainWindow)
        self.actionGradient.setObjectName("actionGradient")
        self.actionConvex = QtWidgets.QAction(MainWindow)
        self.actionConvex.setObjectName("actionConvex")
        self.actionx_direcction_Sobel = QtWidgets.QAction(MainWindow)
        self.actionx_direcction_Sobel.setObjectName("actionx_direcction_Sobel")
        self.actiony_direction_Sobel = QtWidgets.QAction(MainWindow)
        self.actiony_direction_Sobel.setObjectName("actiony_direction_Sobel")
        self.actionLaplacian = QtWidgets.QAction(MainWindow)
        self.actionLaplacian.setObjectName("actionLaplacian")
        self.actionLaplacian_of_Gaussian = QtWidgets.QAction(MainWindow)
        self.actionLaplacian_of_Gaussian.setObjectName("actionLaplacian_of_Gaussian")
        self.actionHough = QtWidgets.QAction(MainWindow)
        self.actionHough.setObjectName("actionHough")
        self.actionSmall = QtWidgets.QAction(MainWindow)
        self.actionSmall.setObjectName("actionSmall")
        self.actionRotation = QtWidgets.QAction(MainWindow)
        self.actionRotation.setObjectName("actionRotation")
        self.actionAffine = QtWidgets.QAction(MainWindow)
        self.actionAffine.setObjectName("actionAffine")
        self.actionGamma = QtWidgets.QAction(MainWindow)
        self.actionGamma.setObjectName("actionGamma")
        self.actionBig = QtWidgets.QAction(MainWindow)
        self.actionBig.setObjectName("actionBig")
        self.actionQt = QtWidgets.QAction(MainWindow)
        self.actionQt.setObjectName("actionQt")
        self.actionAuthor = QtWidgets.QAction(MainWindow)
        self.actionAuthor.setObjectName("actionAuthor")
        self.actionTranslation = QtWidgets.QAction(MainWindow)
        self.actionTranslation.setObjectName("actionTranslation")
        self.actionGaussian = QtWidgets.QAction(MainWindow)
        self.actionGaussian.setObjectName("actionGaussian")
        self.actionRayleigh = QtWidgets.QAction(MainWindow)
        self.actionRayleigh.setObjectName("actionRayleigh")
        self.actionImpluse = QtWidgets.QAction(MainWindow)
        self.actionImpluse.setObjectName("actionImpluse")
        self.actionUniform = QtWidgets.QAction(MainWindow)
        self.actionUniform.setObjectName("actionUniform")
        self.actionErlang = QtWidgets.QAction(MainWindow)
        self.actionErlang.setObjectName("actionErlang")
        self.actionHistogram_PDF = QtWidgets.QAction(MainWindow)
        self.actionHistogram_PDF.setObjectName("actionHistogram_PDF")
        self.actionDetecion = QtWidgets.QAction(MainWindow)
        self.actionDetecion.setObjectName("actionDetecion")
        self.actionHoughLines = QtWidgets.QAction(MainWindow)
        self.actionHoughLines.setObjectName("actionHoughLines")
        self.actionSHT = QtWidgets.QAction(MainWindow)
        self.actionSHT.setObjectName("actionSHT")
        self.actionMedian_Filtering = QtWidgets.QAction(MainWindow)
        self.actionMedian_Filtering.setObjectName("actionMedian_Filtering")
        self.actionAdaptive_Wiener_Filtering = QtWidgets.QAction(MainWindow)
        self.actionAdaptive_Wiener_Filtering.setObjectName("actionAdaptive_Wiener_Filtering")
        self.actionAdaptive_Median_Filtering = QtWidgets.QAction(MainWindow)
        self.actionAdaptive_Median_Filtering.setObjectName("actionAdaptive_Median_Filtering")
        self.actionInverse_Filter = QtWidgets.QAction(MainWindow)
        self.actionInverse_Filter.setObjectName("actionInverse_Filter")
        self.actionBlur = QtWidgets.QAction(MainWindow)
        self.actionBlur.setObjectName("actionBlur")
        self.actionBox_Filter = QtWidgets.QAction(MainWindow)
        self.actionBox_Filter.setObjectName("actionBox_Filter")
        self.actionMedian_Filter = QtWidgets.QAction(MainWindow)
        self.actionMedian_Filter.setObjectName("actionMedian_Filter")
        self.actionBilateral_Filter = QtWidgets.QAction(MainWindow)
        self.actionBilateral_Filter.setObjectName("actionBilateral_Filter")
        self.actionGaussian_Filter = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Filter.setObjectName("actionGaussian_Filter")
        self.actionDirectional_Filtering = QtWidgets.QAction(MainWindow)
        self.actionDirectional_Filtering.setObjectName("actionDirectional_Filtering")
        self.actionMedian_threshold = QtWidgets.QAction(MainWindow)
        self.actionMedian_threshold.setObjectName("actionMedian_threshold")
        self.actionMedian_threshold = QtWidgets.QAction(MainWindow)
        self.actionMedian_threshold.setObjectName("actionMedian_threshold")
        self.actionDirectional_Filtering = QtWidgets.QAction(MainWindow)
        self.actionDirectional_Filtering.setObjectName("actionDirectional_Filtering")
        self.action_Butterworth = QtWidgets.QAction(MainWindow)
        self.action_Butterworth.setObjectName("action_Butterworth")
        self.action_Notch_filter = QtWidgets.QAction(MainWindow)
        self.action_Notch_filter.setObjectName("action_Notch_filter")
        self.actionCartoon = QtWidgets.QAction(MainWindow)
        self.actionCartoon.setObjectName("actionCartoon")
        self.actionDirectional_Filtering_3 = QtWidgets.QAction(MainWindow)
        self.actionDirectional_Filtering_3.setObjectName("actionDirectional_Filtering_3")
        self.actionDirectional_Filtering_4 = QtWidgets.QAction(MainWindow)
        self.actionDirectional_Filtering_4.setObjectName("actionDirectional_Filtering_4")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.reset_button.setText(_translate("MainWindow", "RESET"))
        self.ok_button.setText(_translate("MainWindow", "OK"))
        self.label_14.setText(_translate("MainWindow", "Edge detection"))
        self.label_15.setText(_translate("MainWindow", "Min"))
        self.label_16.setText(_translate("MainWindow", "Max"))
        self.label_17.setText(_translate("MainWindow", "Face detection"))
        self.label_18.setText(_translate("MainWindow", "ScaleFactor"))
        self.label_10.setText(_translate("MainWindow", "Properties"))
        self.label_7.setText(_translate("MainWindow", "Bilateral"))
        self.label_8.setText(_translate("MainWindow", "Median"))
        self.label_11.setText(_translate("MainWindow", "Normalized"))
        self.label_12.setText(_translate("MainWindow", "Gaussian"))
        self.label_13.setText(_translate("MainWindow", "Filtering"))
        self.filter_checkbox.setText(_translate("MainWindow", "Filtering"))
        self.edge_checkbox.setText(_translate("MainWindow", "Edge detection"))
        self.face_detect_checkbox.setText(_translate("MainWindow", "Face detection"))
        self.face_recog_checkbox.setText(_translate("MainWindow", "Face recognition"))
        self.before_label.setText(_translate("MainWindow", "Before"))
        self.after_label.setText(_translate("MainWindow", "After"))
        self.open_button.setText(_translate("MainWindow", "Open"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.export_button.setText(_translate("MainWindow", "Export"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionPrint.setText(_translate("MainWindow", "Print"))
        self.actionQuit.setText(_translate("MainWindow", "Exit"))
        self.actioAnhXam.setText(_translate("MainWindow", "Graycsale"))
        self.actionCanny.setText(_translate("MainWindow", "Canny"))
        self.actionNegative.setText(_translate("MainWindow", "Negative"))
        self.actionHistogram.setText(_translate("MainWindow", "Histogram Equal"))
        self.actionLog.setText(_translate("MainWindow", "Log "))
        self.actionGaussan.setText(_translate("MainWindow", "Gaussan"))
        self.actionHigh_Boost.setText(_translate("MainWindow", "High Boost"))
        self.actionLaplacian.setText(_translate("MainWindow", "Laplacian"))
        self.actionFilter_Average.setText(_translate("MainWindow", "Median"))
        self.actionUnsharp.setText(_translate("MainWindow", "Unsharp"))
        self.actionCh_ng_5.setText(_translate("MainWindow", "Chương 5"))
        self.actionTanSo.setText(_translate("MainWindow", "Ảnh Miền Tần Số"))
        self.actionIdeal_LPF.setText(_translate("MainWindow", "Ideal LPF"))
        self.actionButter_LPF.setText(_translate("MainWindow", "Butterworth LPF"))
        self.actionGaussian_LPF.setText(_translate("MainWindow", "Gaussian LPF"))
        self.actionIdeal_HPF.setText(_translate("MainWindow", "Ideal HPF"))
        self.actionButterworth_HPF.setText(_translate("MainWindow", "Butterworth HPF"))
        self.actionGaussian_HPF.setText(_translate("MainWindow", "Gaussian HPF"))
        self.actiondilate.setText(_translate("MainWindow", "dilate"))
        self.actionErode.setText(_translate("MainWindow", "Erode"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionHit_miss.setText(_translate("MainWindow", "Hit-miss"))
        self.actionDilate.setText(_translate("MainWindow", "Dilate"))
        self.actionMorboundary.setText(_translate("MainWindow", "Morboundary"))
        self.actionGradient.setText(_translate("MainWindow", "Gradient"))
        self.actionConvex.setText(_translate("MainWindow", "Convex"))
        self.actionx_direcction_Sobel.setText(_translate("MainWindow", "Sobel X"))
        self.actiony_direction_Sobel.setText(_translate("MainWindow", "Sobel Y"))
        self.actionLaplacian.setText(_translate("MainWindow", "Sobel Laplacian"))
        self.actionLaplacian_of_Gaussian.setText(_translate("MainWindow", "Laplacian of Gaussian"))
        self.actionHough.setText(_translate("MainWindow", "Hough"))
        self.actionSmall.setText(_translate("MainWindow", "Zoom in"))
        self.actionRotation.setText(_translate("MainWindow", "Rotation"))
        self.actionAffine.setText(_translate("MainWindow", "Shearing"))
        self.actionGamma.setText(_translate("MainWindow", "Gamma"))
        self.actionBig.setText(_translate("MainWindow", "Zoom out"))
        self.actionQt.setText(_translate("MainWindow", "About Qt"))
        self.actionAuthor.setText(_translate("MainWindow", "Author"))
        self.actionTranslation.setText(_translate("MainWindow", "Translation"))
        self.actionGaussian.setText(_translate("MainWindow", "Gaussian"))
        self.actionRayleigh.setText(_translate("MainWindow", "Rayleigh"))
        self.actionImpluse.setText(_translate("MainWindow", "Impluse"))
        self.actionUniform.setText(_translate("MainWindow", "Uniform"))
        self.actionErlang.setText(_translate("MainWindow", "Erlang"))
        self.actionHistogram_PDF.setText(_translate("MainWindow", "Histogram PDF"))
        self.actionDetecion.setText(_translate("MainWindow", "HoughLines"))
        self.actionHoughLines.setText(_translate("MainWindow", "HoughLines"))
        self.actionSHT.setText(_translate("MainWindow", " Standard Hough Transform"))
        self.actionMedian_Filtering.setText(_translate("MainWindow", "Median Filtering"))
        self.actionAdaptive_Wiener_Filtering.setText(_translate("MainWindow", "Adaptive Wiener Filtering"))
        self.actionAdaptive_Median_Filtering.setText(_translate("MainWindow", "Adaptive Median Filtering"))
        self.actionInverse_Filter.setText(_translate("MainWindow", "Inverse Filter "))
        self.actionBlur.setText(_translate("MainWindow", "Blur"))
        self.actionBox_Filter.setText(_translate("MainWindow", "Box Filter"))
        self.actionMedian_Filter.setText(_translate("MainWindow", "Median Filter"))
        self.actionBilateral_Filter.setText(_translate("MainWindow", "Bilateral Filter"))
        self.actionGaussian_Filter.setText(_translate("MainWindow", "Gaussian Filter"))
        self.actionDirectional_Filtering.setText(_translate("MainWindow", "Directional Filtering"))
        self.actionMedian_threshold.setText(_translate("MainWindow", "Median threshold"))
        self.actionMedian_threshold.setText(_translate("MainWindow", "Median threshold"))
        self.actionDirectional_Filtering.setText(_translate("MainWindow", "Directional Filtering"))
        self.action_Butterworth.setText(_translate("MainWindow", "Butterworth Filter"))
        self.action_Notch_filter.setText(_translate("MainWindow", "Notch Filter"))
        self.actionCartoon.setText(_translate("MainWindow", "Cartoon"))
        self.actionDirectional_Filtering_3.setText(_translate("MainWindow", "Directional Filtering 2"))
        self.actionDirectional_Filtering_4.setText(_translate("MainWindow", "Directional Filtering 3"))

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
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def open_img(self):
        # fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\DELL\PycharmProjects\DemoPro', "Image Files (*)")
        # if fname:
        #     self.loadImage(fname)
        # else:
        #     print("Invalid Image")
        print("cc")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())