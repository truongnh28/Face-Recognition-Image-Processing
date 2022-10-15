# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1440, 876)
        MainWindow.setStyleSheet("background-color:rgb(255,255,255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.properties_frame = QtWidgets.QFrame(self.centralwidget)
        self.properties_frame.setGeometry(QtCore.QRect(1070, 10, 351, 691))
        self.properties_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.properties_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.properties_frame.setObjectName("properties_frame")
        self.resetButton = QtWidgets.QPushButton(self.properties_frame)
        self.resetButton.setGeometry(QtCore.QRect(40, 590, 91, 41))
        self.resetButton.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.resetButton.setObjectName("resetButton")
        self.okButton = QtWidgets.QPushButton(self.properties_frame)
        self.okButton.setGeometry(QtCore.QRect(200, 590, 91, 41))
        self.okButton.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.okButton.setObjectName("okButton")
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
        self.edge_detact_min_param.setGeometry(QtCore.QRect(120, 40, 201, 22))
        self.edge_detact_min_param.setMinimum(1)
        self.edge_detact_min_param.setMaximum(10)
        self.edge_detact_min_param.setProperty("value", 1)
        self.edge_detact_min_param.setSliderPosition(1)
        self.edge_detact_min_param.setOrientation(QtCore.Qt.Horizontal)
        self.edge_detact_min_param.setObjectName("edge_detact_min_param")
        self.edge_detact_max_param = QtWidgets.QSlider(self.edge_detection_frame)
        self.edge_detact_max_param.setGeometry(QtCore.QRect(120, 70, 201, 22))
        self.edge_detact_max_param.setMinimum(1)
        self.edge_detact_max_param.setMaximum(10)
        self.edge_detact_max_param.setProperty("value", 1)
        self.edge_detact_max_param.setSliderPosition(1)
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
        self.scale_factor_param.setGeometry(QtCore.QRect(120, 60, 201, 22))
        self.scale_factor_param.setMinimum(1)
        self.scale_factor_param.setMaximum(10)
        self.scale_factor_param.setProperty("value", 1)
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
        self.bilateral_param_2 = QtWidgets.QSlider(self.filtering_frame)
        self.bilateral_param_2.setGeometry(QtCore.QRect(120, 140, 201, 22))
        self.bilateral_param_2.setMinimum(0)
        self.bilateral_param_2.setMaximum(4)
        self.bilateral_param_2.setPageStep(10)
        self.bilateral_param_2.setProperty("value", 0)
        self.bilateral_param_2.setSliderPosition(0)
        self.bilateral_param_2.setOrientation(QtCore.Qt.Horizontal)
        self.bilateral_param_2.setObjectName("bilateral_param_2")
        self.gaussian_param_2 = QtWidgets.QSlider(self.filtering_frame)
        self.gaussian_param_2.setGeometry(QtCore.QRect(120, 50, 201, 22))
        self.gaussian_param_2.setMinimum(1)
        self.gaussian_param_2.setMaximum(10)
        self.gaussian_param_2.setProperty("value", 1)
        self.gaussian_param_2.setSliderPosition(1)
        self.gaussian_param_2.setOrientation(QtCore.Qt.Horizontal)
        self.gaussian_param_2.setObjectName("gaussian_param_2")
        self.normalized_param_2 = QtWidgets.QSlider(self.filtering_frame)
        self.normalized_param_2.setGeometry(QtCore.QRect(120, 80, 201, 22))
        self.normalized_param_2.setMinimum(5)
        self.normalized_param_2.setMaximum(20)
        self.normalized_param_2.setProperty("value", 5)
        self.normalized_param_2.setSliderPosition(5)
        self.normalized_param_2.setOrientation(QtCore.Qt.Horizontal)
        self.normalized_param_2.setObjectName("normalized_param_2")
        self.median_param_2 = QtWidgets.QSlider(self.filtering_frame)
        self.median_param_2.setGeometry(QtCore.QRect(120, 110, 201, 22))
        self.median_param_2.setMinimum(0)
        self.median_param_2.setMaximum(4)
        self.median_param_2.setPageStep(10)
        self.median_param_2.setProperty("value", 0)
        self.median_param_2.setSliderPosition(0)
        self.median_param_2.setOrientation(QtCore.Qt.Horizontal)
        self.median_param_2.setObjectName("median_param_2")
        self.label_13 = QtWidgets.QLabel(self.filtering_frame)
        self.label_13.setGeometry(QtCore.QRect(10, 10, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.main_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_frame.setGeometry(QtCore.QRect(19, 10, 1041, 701))
        self.main_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_frame.setObjectName("main_frame")
        self.image_before = QtWidgets.QLabel(self.main_frame)
        self.image_before.setGeometry(QtCore.QRect(10, 50, 481, 491))
        self.image_before.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.image_before.setFrameShape(QtWidgets.QFrame.Panel)
        self.image_before.setFrameShadow(QtWidgets.QFrame.Plain)
        self.image_before.setText("")
        self.image_before.setObjectName("image_before")
        self.image_after = QtWidgets.QLabel(self.main_frame)
        self.image_after.setGeometry(QtCore.QRect(520, 50, 511, 491))
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
        self.main_frame.raise_()
        self.properties_frame.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1440, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuCh_ng_6 = QtWidgets.QMenu(self.menubar)
        self.menuCh_ng_6.setObjectName("menuCh_ng_6")
        self.menuCh_ng_9 = QtWidgets.QMenu(self.menubar)
        self.menuCh_ng_9.setObjectName("menuCh_ng_9")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        self.menuType_Noise = QtWidgets.QMenu(self.menubar)
        self.menuType_Noise.setObjectName("menuType_Noise")
        self.menuType_Here = QtWidgets.QMenu(self.menuType_Noise)
        self.menuType_Here.setObjectName("menuType_Here")
        self.menuImage_Restoration_2 = QtWidgets.QMenu(self.menuType_Noise)
        self.menuImage_Restoration_2.setObjectName("menuImage_Restoration_2")
        self.menuImage_Restoration = QtWidgets.QMenu(self.menuType_Noise)
        self.menuImage_Restoration.setObjectName("menuImage_Restoration")
        self.menuType_Noise_3 = QtWidgets.QMenu(self.menuImage_Restoration)
        self.menuType_Noise_3.setObjectName("menuType_Noise_3")
        self.menuSimple_Edge_Detection_2 = QtWidgets.QMenu(self.menubar)
        self.menuSimple_Edge_Detection_2.setObjectName("menuSimple_Edge_Detection_2")
        self.menuSmoothing = QtWidgets.QMenu(self.menubar)
        self.menuSmoothing.setObjectName("menuSmoothing")
        self.menuFilter = QtWidgets.QMenu(self.menubar)
        self.menuFilter.setObjectName("menuFilter")
        self.menuCartooning_of_an_Image = QtWidgets.QMenu(self.menubar)
        self.menuCartooning_of_an_Image.setObjectName("menuCartooning_of_an_Image")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../Image-Processing-with-OpenCV-Python-and-PyQt5/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../Image-Processing-with-OpenCV-Python-and-PyQt5/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon1)
        self.actionSave.setObjectName("actionSave")
        self.actionPrint = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../Image-Processing-with-OpenCV-Python-and-PyQt5/print.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPrint.setIcon(icon2)
        self.actionPrint.setObjectName("actionPrint")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../Image-Processing-with-OpenCV-Python-and-PyQt5/Quit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionQuit.setIcon(icon3)
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
        self.actionOpen_2 = QtWidgets.QAction(MainWindow)
        self.actionOpen_2.setObjectName("actionOpen_2")
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
        self.actionLaplacian_2 = QtWidgets.QAction(MainWindow)
        self.actionLaplacian_2.setObjectName("actionLaplacian_2")
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
        self.actionMedian_threshold_2 = QtWidgets.QAction(MainWindow)
        self.actionMedian_threshold_2.setObjectName("actionMedian_threshold_2")
        self.actionDirectional_Filtering_2 = QtWidgets.QAction(MainWindow)
        self.actionDirectional_Filtering_2.setObjectName("actionDirectional_Filtering_2")
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
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionPrint)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuView.addAction(self.actionSmall)
        self.menuView.addAction(self.actionBig)
        self.menuCh_ng_6.addAction(self.actionRotation)
        self.menuCh_ng_6.addAction(self.actionAffine)
        self.menuCh_ng_6.addAction(self.actionTranslation)
        self.menuCh_ng_9.addAction(self.actioAnhXam)
        self.menuCh_ng_9.addAction(self.actionNegative)
        self.menuCh_ng_9.addAction(self.actionLog)
        self.menuCh_ng_9.addAction(self.actionHistogram)
        self.menuCh_ng_9.addAction(self.actionGamma)
        self.menuAbout.addAction(self.actionQt)
        self.menuAbout.addAction(self.actionAuthor)
        self.menuType_Here.addAction(self.actionMedian_Filtering)
        self.menuType_Here.addAction(self.actionAdaptive_Median_Filtering)
        self.menuType_Here.addAction(self.actionAdaptive_Wiener_Filtering)
        self.menuImage_Restoration_2.addAction(self.actionInverse_Filter)
        self.menuType_Noise_3.addAction(self.actionGaussian)
        self.menuType_Noise_3.addAction(self.actionRayleigh)
        self.menuType_Noise_3.addAction(self.actionErlang)
        self.menuType_Noise_3.addAction(self.actionUniform)
        self.menuType_Noise_3.addAction(self.actionImpluse)
        self.menuImage_Restoration.addAction(self.menuType_Noise_3.menuAction())
        self.menuImage_Restoration.addSeparator()
        self.menuImage_Restoration.addAction(self.actionHistogram_PDF)
        self.menuType_Noise.addSeparator()
        self.menuType_Noise.addAction(self.menuImage_Restoration.menuAction())
        self.menuType_Noise.addAction(self.menuType_Here.menuAction())
        self.menuType_Noise.addAction(self.menuImage_Restoration_2.menuAction())
        self.menuSimple_Edge_Detection_2.addAction(self.actionSHT)
        self.menuSmoothing.addAction(self.actionBlur)
        self.menuSmoothing.addAction(self.actionBox_Filter)
        self.menuSmoothing.addAction(self.actionMedian_Filter)
        self.menuSmoothing.addAction(self.actionBilateral_Filter)
        self.menuSmoothing.addAction(self.actionGaussian_Filter)
        self.menuFilter.addAction(self.actionMedian_threshold_2)
        self.menuFilter.addAction(self.actionDirectional_Filtering_2)
        self.menuFilter.addAction(self.actionDirectional_Filtering_3)
        self.menuFilter.addAction(self.actionDirectional_Filtering_4)
        self.menuFilter.addAction(self.action_Butterworth)
        self.menuFilter.addAction(self.action_Notch_filter)
        self.menuCartooning_of_an_Image.addAction(self.actionCartoon)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuCh_ng_6.menuAction())
        self.menubar.addAction(self.menuCh_ng_9.menuAction())
        self.menubar.addAction(self.menuSimple_Edge_Detection_2.menuAction())
        self.menubar.addAction(self.menuCartooning_of_an_Image.menuAction())
        self.menubar.addAction(self.menuSmoothing.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())
        self.menubar.addAction(self.menuType_Noise.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionPrint)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionQuit)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.resetButton.setText(_translate("MainWindow", "RESET"))
        self.okButton.setText(_translate("MainWindow", "OK"))
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
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuCh_ng_6.setTitle(_translate("MainWindow", "Chapter 2"))
        self.menuCh_ng_9.setTitle(_translate("MainWindow", "Chapter 3"))
        self.menuAbout.setTitle(_translate("MainWindow", "About US"))
        self.menuType_Noise.setTitle(_translate("MainWindow", "Image Restoration\n"
""))
        self.menuType_Here.setTitle(_translate("MainWindow", "Image Restoration 1\n"
""))
        self.menuImage_Restoration_2.setTitle(_translate("MainWindow", "Image Restoration 2"))
        self.menuImage_Restoration.setTitle(_translate("MainWindow", "Image Restoration "))
        self.menuType_Noise_3.setTitle(_translate("MainWindow", "Type Noise"))
        self.menuSimple_Edge_Detection_2.setTitle(_translate("MainWindow", "Simple Edge Detection"))
        self.menuSmoothing.setTitle(_translate("MainWindow", "Smoothing"))
        self.menuFilter.setTitle(_translate("MainWindow", "Filter"))
        self.menuCartooning_of_an_Image.setTitle(_translate("MainWindow", "Cartooning of an Image"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
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
        self.actionOpen_2.setText(_translate("MainWindow", "Open"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionHit_miss.setText(_translate("MainWindow", "Hit-miss"))
        self.actionDilate.setText(_translate("MainWindow", "Dilate"))
        self.actionMorboundary.setText(_translate("MainWindow", "Morboundary"))
        self.actionGradient.setText(_translate("MainWindow", "Gradient"))
        self.actionConvex.setText(_translate("MainWindow", "Convex"))
        self.actionx_direcction_Sobel.setText(_translate("MainWindow", "Sobel X"))
        self.actiony_direction_Sobel.setText(_translate("MainWindow", "Sobel Y"))
        self.actionLaplacian_2.setText(_translate("MainWindow", "Sobel Laplacian"))
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
        self.actionMedian_threshold_2.setText(_translate("MainWindow", "Median threshold"))
        self.actionDirectional_Filtering_2.setText(_translate("MainWindow", "Directional Filtering"))
        self.action_Butterworth.setText(_translate("MainWindow", "Butterworth Filter"))
        self.action_Notch_filter.setText(_translate("MainWindow", "Notch Filter"))
        self.actionCartoon.setText(_translate("MainWindow", "Cartoon"))
        self.actionDirectional_Filtering_3.setText(_translate("MainWindow", "Directional Filtering 2"))
        self.actionDirectional_Filtering_4.setText(_translate("MainWindow", "Directional Filtering 3"))
