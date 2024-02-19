from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QStyle
from passporteye import read_mrz
import sqlite3
import os
import sys
import filetype
import cv2
import dlib
import numpy as np
import pandas as pd
import webbrowser
from numpy import dot
from numpy.linalg import norm
from report_generator import generate_report
import time
import torch
from PIL import Image
torch.autograd.set_grad_enabled(False)
from torchvision import transforms
from mtlface.face_aligment import face_process


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, MainWindow):
        super().__init__()
        self.setupUi(MainWindow=MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        MainWindow.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
        MainWindow.setWindowIcon(QtGui.QIcon(QPixmap("Resources/logo/COMSATS_logo.png")))
        MainWindow.resize(1280, 780)
        MainWindow.setMinimumSize(QtCore.QSize(1280, 768))
        MainWindow.setMaximumSize(QtCore.QSize(1280, 768))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setMouseTracking(False)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setStyleSheet("""
            QPushButton{
                color:rgb(17,17,17);
                border-width: 1px;
                border-radius: 6px;
                border-bottom-color: rgb(150,150,150);
                border-right-color: rgb(165,165,165);
                border-left-color: rgb(165,165,165);
                border-top-color: rgb(180,180,180);
                border-style: solid;
                padding: 4px;
                background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
            }
            QPushButton:hover{
                color: rgb(0, 204, 0);
                border-width: 1px;
                border-radius:6px;
                border-top-color: rgb(0, 204, 0);
                border-right-color: rgb(0, 204, 0);
                border-left-color:  rgb(0, 204, 0);
                border-bottom-color: rgb(0, 204, 0);
                border-style: solid;
                padding: 2px;
                font-weight: bold;
                background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
            }
            QPushButton:default{
                color:rgb(17,17,17);
                border-width: 1px;
                border-radius:6px;
                border-top-color: rgb(255,150,60);
                border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 255));
                border-left-color:  qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 255));
                border-bottom-color: rgb(200,70,20);
                border-style: solid;
                padding: 2px;
                background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
            }
            QPushButton:pressed{
                color: rgb(175, 100, 100);
                border-width: 3px;
                border-radius: 6px;
                border-width: 1px;
                border-top-color: rgb(175, 100, 100);
                border-right-color: rgb(175, 100, 100);
                border-left-color:  rgb(175, 100, 100);
                border-bottom-color: rgb(175, 100, 100);
                border-style: solid;
                font-weight: bold;
                padding: 2px;
                background-color: transparent;
            }
            QPushButton:disabled{
                color:rgb(174,167,159);
                border-width: 1px;
                border-radius: 6px;
                background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(200, 200, 200, 255), stop:1 rgba(230, 230, 230, 255));
            }   
        """)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("""
            QWidget {
                background-color:rgb(235,235,235);
            }
        """)
        self.centralwidget.setObjectName("centralwidget")
        self.projlabel = QtWidgets.QLabel(self.centralwidget)
        self.projlabel.setGeometry(QtCore.QRect(0, 0, 1280, 90))
        self.projlabel.setMinimumSize(QtCore.QSize(1280, 90))
        self.projlabel.setMaximumSize(QtCore.QSize(1366, 90))
        self.projlabel.setBaseSize(QtCore.QSize(1, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.projlabel.setFont(font)
        self.projlabel.setStyleSheet("""
            QLabel {
                background-color:rgb(255, 255, 255);
            };
            background-image: url('Resources/logo/flag.png');
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
            color: white;
        """)
        self.projlabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.projlabel.setObjectName("projlabel")
        self.close_btn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.close_btn.setGeometry(QtCore.QRect(1255, 5, 21, 20))
        self.close_btn.setObjectName("close_btn")
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setFamily("Arial")
        self.close_btn.setFont(font)
        self.close_btn.setStyleSheet("""
            QPushButton#close_btn{
                color: rgb(17,17,17);
                border-width: 1px;
                border-radius: 6px;
                border-bottom-color: rgb(150,150,150);
                border-right-color: rgb(165,165,165);
                border-left-color: rgb(165,165,165);
                border-top-color: rgb(180,180,180);
                border-style: solid;
                padding: 4px;
                background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
            }
            QPushButton#close_btn:hover{
                color: rgb(17, 17, 17);
                border-width: 2px;
                border-radius:6px;
                border-top-color: rgb(255, 0, 0);
                border-right-color: rgb(255, 0, 0);
                border-left-color:  rgb(255, 0, 0);
                border-bottom-color: rgb(255, 0, 0);
                border-style: solid;
                padding: 2px;
                font-weight: bold;
                background-color: red;
            }
            QPushButton#close_btn:pressed{
                color: rgb(17, 17, 17);
                border-width: 3px;
                border-radius: 6px;
                border-width: 1px;
                border-top-color: rgb(175, 100, 100);
                border-right-color: rgb(175, 100, 100);
                border-left-color:  rgb(175, 100, 100);
                border-bottom-color: rgb(175, 100, 100);
                border-style: solid;
                font-weight: bold;
                padding: 2px;
                background-color: rgb(175, 100, 100);
            }
        """)
        pixmapi = QStyle.StandardPixmap.SP_TitleBarCloseButton
        icon = self.style().standardIcon(pixmapi)
        self.close_btn.setIcon(icon)
        self.close_btn.setWhatsThis("Close Application")
        self.close_btn.setToolTip("Close Application")
        self.min_btn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.min_btn.setGeometry(QtCore.QRect(1225, 5, 21, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.min_btn.setFont(font)
        pixmapi = QStyle.StandardPixmap.SP_TitleBarMinButton
        icon = self.style().standardIcon(pixmapi)
        self.min_btn.setIcon(icon)
        self.min_btn.setObjectName("min_btn")
        self.min_btn.setToolTip("Minimize Window")
        self.min_btn.setWhatsThis("Minimize Window")
        self.min_btn.setStyleSheet("""
            QPushButton#win_btn{
                color: rgb(17,17,17);
                border-width: 1px;
                border-radius: 6px;
                border-bottom-color: rgb(150,150,150);
                border-right-color: rgb(165,165,165);
                border-left-color: rgb(165,165,165);
                border-top-color: rgb(180,180,180);
                border-style: solid;
                padding: 4px;
                background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
            }
            QPushButton#min_btn:hover{
                color: rgb(17, 17, 17);
                border-width: 2px;
                border-radius:6px;
                border-top-color: rgb(0, 204, 0);
                border-right-color: rgb(0, 204, 0);
                border-left-color:  rgb(0, 204, 0);
                border-bottom-color: rgb(0, 204, 0);
                border-style: solid;
                padding: 2px;
                font-weight: bold;
                background-color: rgb(0, 204, 0);
            }
            QPushButton#min_btn:pressed{
                color: rgb(17, 17, 17);
                border-width: 3px;
                border-radius: 6px;
                border-width: 1px;
                border-top-color: rgb(175, 100, 100);
                border-right-color: rgb(175, 100, 100);
                border-left-color:  rgb(175, 100, 100);
                border-bottom-color: rgb(175, 100, 100);
                border-style: solid;
                font-weight: bold;
                padding: 2px;
                background-color: rgb(175, 100, 100);
            }
        """)
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(600, 120, 280, 420))
        self.groupBox_4.setMinimumSize(QtCore.QSize(150, 420))
        self.groupBox_4.setMaximumSize(QtCore.QSize(320, 370))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.groupBox_4.setAutoFillBackground(False)
        self.groupBox_4.setStyleSheet("""
            QGroupBox {
                background-color:rgb(245,245,245);
                border: 2px solid gray;
            }
        """)
        self.groupBox_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox_4.setObjectName("groupBox_4")
        self.progressBar = QtWidgets.QProgressBar(parent=self.groupBox_4)
        self.progressBar.setGeometry(QtCore.QRect(20, 380, 240, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.progressBar.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.progressBar.setAutoFillBackground(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.Direction.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setFont(font)
        self.progressBar.setStyleSheet("""
            QProgressBar {
                text-align: center;
                color: black;
                border: 1px solid gray;
            }
            QProgressBar::chunk {
                background-color: #17ff17;
                border-radius: 15px;
                margin-right: 0.5px;
                width: 3px;
            }
        """)

        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(40, 5, 80, 80))
        self.logo.setMinimumSize(QtCore.QSize(80, 80))
        self.logo.setMaximumSize(QtCore.QSize(80, 80))
        self.logo.setAutoFillBackground(False)
        self.logo.setStyleSheet("""            
            background-color: transparent;
            QLabel {
                border:transparent;
                }
        """)
        self.logo.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.logo.setLineWidth(1)
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap("Resources/logo/COMSATS_logo.png"))
        self.logo.setScaledContents(True)
        self.logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.logo.setWordWrap(True)
        self.logo.setObjectName("logo")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 120, 540, 420))
        self.groupBox.setMinimumSize(QtCore.QSize(540, 420))
        self.groupBox.setMaximumSize(QtCore.QSize(540, 420))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setStyleSheet("""
            QGroupBox {
                background-color:rgb(245,245,245);
                border: 2px solid gray;
            }
        """)
        self.groupBox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.up_passport_btn = QtWidgets.QPushButton(self.groupBox)
        self.up_passport_btn.setGeometry(QtCore.QRect(100, 370, 140, 35))
        self.up_passport_btn.setToolTip("Upload Passenger's Passport")
        font = QtGui.QFont()
        font.setPointSize(11)
        self.up_passport_btn.setFont(font)
        self.up_passport_btn.setObjectName("up_passport_btn")
        self.passport = QtWidgets.QLabel(parent=self.groupBox)
        self.passport.setGeometry(QtCore.QRect(20, 40, 500, 320))
        self.passport.setMinimumSize(QtCore.QSize(500, 320))
        self.passport.setMaximumSize(QtCore.QSize(580, 400))
        self.passport.setAutoFillBackground(False)
        self.passport.setStyleSheet("""
            QLabel {
                background-color:rgb(220,230,220)
            }
        """)
        self.passport.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        self.passport.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.passport.setLineWidth(1)
        self.passport.setText("")
        self.passport.setScaledContents(True)
        self.passport.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.passport.setObjectName("passport")
        self.verify_passport_btn = QtWidgets.QPushButton(self.groupBox)
        self.verify_passport_btn.setGeometry(QtCore.QRect(320, 370, 100, 35))
        self.verify_passport_btn.setToolTip("Verify Passport")
        font = QtGui.QFont()
        font.setPointSize(11)
        self.verify_passport_btn.setFont(font)
        self.verify_passport_btn.setStyleSheet("")
        self.verify_passport_btn.setObjectName("verify_passport_btn")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 560, 541, 75))
        self.groupBox_2.setMinimumSize(QtCore.QSize(540, 75))
        self.groupBox_2.setMaximumSize(QtCore.QSize(620, 70))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.groupBox_2.setAutoFillBackground(False)
        self.groupBox_2.setStyleSheet("""
            QGroupBox {
                background-color:rgb(245,245,245);
                border: 2px solid gray;
            }
        """)
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 27, 170, 35))
        self.label_2.setMinimumSize(QtCore.QSize(120, 35))
        self.label_2.setMaximumSize(QtCore.QSize(220, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("""
            QLabel {
                background-color:rgb(240,240,240)\n"
            }
        """)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.up_pass_no_btn = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.up_pass_no_btn.setGeometry(QtCore.QRect(400, 27, 120, 35))
        self.up_pass_no_btn.setToolTip("Enter Passport Number and Verify")
        font = QtGui.QFont()
        font.setPointSize(11)
        self.up_pass_no_btn.setFont(font)
        self.up_pass_no_btn.setObjectName("up_pass_no_btn")
        self.pass_no_le = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.pass_no_le.setGeometry(QtCore.QRect(190, 27, 200, 35))
        self.pass_no_le.setMinimumSize(QtCore.QSize(200, 35))
        self.pass_no_le.setMaximumSize(QtCore.QSize(210, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.pass_no_le.setFont(font)
        self.pass_no_le.setStyleSheet("""
            QLineEdit {
                background-color:rgb(220,230,220);
            }
        """)
        self.pass_no_le.setObjectName("pass_no_le")
        self.s_status = QtWidgets.QLabel(self.centralwidget)
        self.s_status.setGeometry(QtCore.QRect(40, 650, 540, 80))
        self.s_status.setMinimumSize(QtCore.QSize(540, 80))
        self.s_status.setMaximumSize(QtCore.QSize(5400, 80))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.s_status.setFont(font)
        self.s_status.setAccessibleDescription("")
        self.s_status.setStyleSheet("""
            QLabel {
                background-color:rgb(245,245,245);
                border: 2px solid gray;
            }
        """)
        self.s_status.setText("")
        self.s_status.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.s_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.s_status.setObjectName("s_status")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(900, 120, 340, 420))
        self.groupBox_3.setMinimumSize(QtCore.QSize(340, 400))
        self.groupBox_3.setMaximumSize(QtCore.QSize(320, 500))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.groupBox_3.setAutoFillBackground(False)
        self.groupBox_3.setStyleSheet("""
            QGroupBox {
                background-color:rgb(245,245,245);
                border: 2px solid gray;
            }
        """)
        self.groupBox_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox_3.setObjectName("groupBox_3")
        self.up_person_btn = QtWidgets.QPushButton(self.groupBox_3)
        self.up_person_btn.setGeometry(QtCore.QRect(150, 370, 170, 35))
        self.up_person_btn.setToolTip("Upload Person's Image")
        font = QtGui.QFont()
        font.setPointSize(11)
        self.up_person_btn.setFont(font)
        self.up_person_btn.setObjectName("up_person_btn")
        self.person = QtWidgets.QLabel(parent=self.groupBox_3)
        self.person.setGeometry(QtCore.QRect(20, 40, 300, 320))
        self.person.setMinimumSize(QtCore.QSize(300, 160))
        self.person.setMaximumSize(QtCore.QSize(280, 320))
        self.person.setStyleSheet("""
            QLabel {
                background-color:rgb(220,230,220);
            }
        """)
        self.person.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        self.person.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.person.setText("")
        self.person.setScaledContents(True)
        self.person.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.person.setObjectName("person")
        font = QtGui.QFont()
        font.setPointSize(11)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.open_cam_btn = QtWidgets.QPushButton(self.groupBox_3)
        self.open_cam_btn.setGeometry(QtCore.QRect(20, 370, 110, 35))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.open_cam_btn.setFont(font)
        self.open_cam_btn.setObjectName("open_cam_btn")
        self.open_cam_btn.setToolTip("Open Camera")
        self.label_5 = QtWidgets.QLabel(self.groupBox_4)
        self.label_5.setGeometry(QtCore.QRect(77, 40, 121, 20))
        self.label_5.setMinimumSize(QtCore.QSize(106, 20))
        self.label_5.setMaximumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("""
            QLabel {
                background-color:rgb(240,240,240);
            }
        """)
        self.label_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox_4)
        self.label_6.setGeometry(QtCore.QRect(21, 324, 121, 20))
        self.label_6.setMinimumSize(QtCore.QSize(100, 20))
        self.label_6.setMaximumSize(QtCore.QSize(120, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("""
            QLabel{
                background-color:rgb(240,240,240);
            }
        """)
        self.label_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_8 = QtWidgets.QLabel(parent=self.groupBox_4)
        self.label_8.setGeometry(QtCore.QRect(150, 324, 106, 20))
        self.label_8.setMinimumSize(QtCore.QSize(106, 20))
        self.label_8.setMaximumSize(QtCore.QSize(130, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("""
            QLabel {
                background-color:rgb(240,240,240);
            }
        """)
        self.label_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.passport_face = QtWidgets.QLabel(self.groupBox_4)
        self.passport_face.setGeometry(QtCore.QRect(24, 204, 106, 106))
        self.passport_face.setMinimumSize(QtCore.QSize(106, 106))
        self.passport_face.setMaximumSize(QtCore.QSize(106, 106))
        self.passport_face.setStyleSheet("""
            QLabel {
                background-color:rgb(220,230,220);
                border: 1px solid gray;
            }
        """)
        self.passport_face.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.passport_face.setText("")
        self.passport_face.setScaledContents(True)
        self.passport_face.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.passport_face.setObjectName("passport_face")
        self.db_face = QtWidgets.QLabel(self.groupBox_4)
        self.db_face.setGeometry(QtCore.QRect(85, 70, 106, 106))
        self.db_face.setMinimumSize(QtCore.QSize(106, 106))
        self.db_face.setMaximumSize(QtCore.QSize(112, 112))
        self.db_face.setStyleSheet("""
            QLabel {
                background-color:rgb(220,230,220);
                border: 1px solid gray;
            }
        """)
        self.db_face.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.db_face.setText("")
        self.db_face.setScaledContents(True)
        self.db_face.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.db_face.setObjectName("db_face")
        self.line = QtWidgets.QFrame(parent=self.groupBox_4)
        self.line.setGeometry(QtCore.QRect(46, 90, 188, 4))
        self.line.setMinimumSize(QtCore.QSize(0, 4))
        self.line.setMaximumSize(QtCore.QSize(320, 4))
        self.line.setAccessibleDescription("")
        self.line.setAutoFillBackground(False)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setLineWidth(1)
        self.line.setMidLineWidth(5)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setObjectName("line")
        self.line_4 = QtWidgets.QFrame(parent=self.groupBox_4)
        self.line_4.setGeometry(QtCore.QRect(45, 90, 3, 116))
        self.line_4.setMidLineWidth(5)
        self.line_4.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(parent=self.groupBox_4)
        self.line_5.setGeometry(QtCore.QRect(231, 90, 3, 116))
        self.line_5.setMidLineWidth(5)
        self.line_5.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_5.setObjectName("line_5")
        self.person_face = QtWidgets.QLabel(self.groupBox_4)
        self.person_face.setGeometry(QtCore.QRect(150, 204, 106, 106))
        self.person_face.setMinimumSize(QtCore.QSize(106, 106))
        self.person_face.setMaximumSize(QtCore.QSize(100, 100))
        self.person_face.setStyleSheet("""
            QLabel {
                background-color:rgb(220,230,220);
                border: 1px solid gray;
            }
        """)
        self.person_face.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.person_face.setText("")
        self.person_face.setScaledContents(True)
        self.person_face.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.person_face.setObjectName("person_face")
        self.pass_percentage = QtWidgets.QLabel(parent=self.groupBox_4)
        self.pass_percentage.setGeometry(QtCore.QRect(15, 135, 50, 40))
        self.pass_percentage.setMinimumSize(QtCore.QSize(50, 40))
        self.pass_percentage.setMaximumSize(QtCore.QSize(540, 80))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pass_percentage.setFont(font)
        self.pass_percentage.setAccessibleDescription("")
        self.pass_percentage.setStyleSheet("""
            QLabel{
                background-color:rgb(220,230,220);
                border: 1px solid gray;
            }
        """)
        self.pass_percentage.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.pass_percentage.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pass_percentage.setObjectName("pass_percentage")
        self.pers_percentage = QtWidgets.QLabel(parent=self.groupBox_4)
        self.pers_percentage.setGeometry(QtCore.QRect(210, 135, 50, 40))
        self.pers_percentage.setMinimumSize(QtCore.QSize(50, 40))
        self.pers_percentage.setMaximumSize(QtCore.QSize(540, 80))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pers_percentage.setFont(font)
        self.pers_percentage.setAccessibleDescription("")
        self.pers_percentage.setStyleSheet("""
            QLabel{
                background-color:rgb(220,230,220);
                border: 1px solid gray;
            }
        """)
        self.pers_percentage.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.pers_percentage.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pers_percentage.setObjectName("pers_percentage")
        self.line_2 = QtWidgets.QFrame(parent=self.groupBox_4)
        self.line_2.setGeometry(QtCore.QRect(1, 360, 278, 4))
        self.line_2.setMinimumSize(QtCore.QSize(0, 4))
        self.line_2.setMaximumSize(QtCore.QSize(320, 4))
        self.line_2.setAccessibleDescription("")
        self.line_2.setAutoFillBackground(False)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.line_2.setLineWidth(1)
        self.line_2.setMidLineWidth(2)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setObjectName("line_2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(900, 560, 340, 170))
        self.groupBox_5.setMinimumSize(QtCore.QSize(340, 170))
        self.groupBox_5.setMaximumSize(QtCore.QSize(340, 170))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.groupBox_5.setAutoFillBackground(False)
        self.groupBox_5.setStyleSheet("""
            QGroupBox {
                background-color:rgb(245,245,245);
                border:2px solid gray;
            }
        """)
        self.groupBox_5.setObjectName("groupBox_5")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.groupBox_5)
        self.plainTextEdit.setGeometry(QtCore.QRect(20, 30, 300, 120))
        self.plainTextEdit.setMinimumSize(QtCore.QSize(300, 120))
        self.plainTextEdit.setMaximumSize(QtCore.QSize(300, 100))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setStyleSheet("""
            QPlainTextEdit{
                background-color:rgb(220,230,220)
            }
        """)
        self.plainTextEdit.setUndoRedoEnabled(False)
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.groupBox_6 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(600, 560, 280, 170))
        self.groupBox_6.setMinimumSize(QtCore.QSize(280, 130))
        self.groupBox_6.setMaximumSize(QtCore.QSize(560, 230))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.groupBox_6.setAutoFillBackground(False)
        self.groupBox_6.setStyleSheet("""
            QGroupBox {
                background-color:rgb(245,245,245);
                border: 2px solid gray;
            }
        """)
        self.groupBox_6.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox_6.setObjectName("groupBox_6")
        self.db_name = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.db_name.setGeometry(QtCore.QRect(110, 30, 150, 30))
        self.db_name.setMinimumSize(QtCore.QSize(150, 30))
        self.db_name.setMaximumSize(QtCore.QSize(220, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.db_name.setFont(font)
        self.db_name.setMouseTracking(False)
        self.db_name.setStyleSheet("""
            QLineEdit {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.db_name.setFrame(False)
        self.db_name.setReadOnly(True)
        self.db_name.setObjectName("db_name")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_6)
        self.label_3.setGeometry(QtCore.QRect(20, 30, 100, 30))
        self.label_3.setMinimumSize(QtCore.QSize(100, 30))
        self.label_3.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("""
            QLabel {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.label_3.setScaledContents(True)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_6)
        self.label_4.setGeometry(QtCore.QRect(20, 60, 100, 30))
        self.label_4.setMinimumSize(QtCore.QSize(100, 30))
        self.label_4.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("""
            QLabel {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.db_passNo = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.db_passNo.setGeometry(QtCore.QRect(110, 60, 150, 30))
        self.db_passNo.setMinimumSize(QtCore.QSize(150, 30))
        self.db_passNo.setMaximumSize(QtCore.QSize(220, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.db_passNo.setFont(font)
        self.db_passNo.setMouseTracking(False)
        self.db_passNo.setStyleSheet("""
            QLineEdit {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.db_passNo.setFrame(False)
        self.db_passNo.setReadOnly(True)
        self.db_passNo.setObjectName("db_passNo")
        self.label_7 = QtWidgets.QLabel(parent=self.groupBox_6)
        self.label_7.setGeometry(QtCore.QRect(20, 90, 100, 30))
        self.label_7.setMinimumSize(QtCore.QSize(100, 30))
        self.label_7.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("""
            QLabel {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.label_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.db_gender = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.db_gender.setGeometry(QtCore.QRect(110, 90, 150, 30))
        self.db_gender.setMinimumSize(QtCore.QSize(150, 30))
        self.db_gender.setMaximumSize(QtCore.QSize(220, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.db_gender.setFont(font)
        self.db_gender.setMouseTracking(False)
        self.db_gender.setStyleSheet("""
            QLineEdit {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.db_gender.setFrame(False)
        self.db_gender.setReadOnly(True)
        self.db_gender.setObjectName("db_gender")
        self.label_9 = QtWidgets.QLabel(parent=self.groupBox_6)
        self.label_9.setGeometry(QtCore.QRect(20, 120, 100, 30))
        self.label_9.setMinimumSize(QtCore.QSize(100, 30))
        self.label_9.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("""
            QLabel {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.label_9.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.db_nationality = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.db_nationality.setGeometry(QtCore.QRect(110, 120, 150, 30))
        self.db_nationality.setMinimumSize(QtCore.QSize(150, 30))
        self.db_nationality.setMaximumSize(QtCore.QSize(220, 35))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.db_nationality.setFont(font)
        self.db_nationality.setMouseTracking(False)
        self.db_nationality.setStyleSheet("""
            QLineEdit {
                background-color:rgb(230,230,230);
                border: 1px solid gray;
            }
        """)
        self.db_nationality.setFrame(False)
        self.db_nationality.setReadOnly(True)
        self.db_nationality.setObjectName("db_nationality")
        self.report_gen_btn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.report_gen_btn.setGeometry(QtCore.QRect(1145, 95, 95, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.report_gen_btn.setFont(font)
        self.report_gen_btn.setAutoDefault(False)
        self.report_gen_btn.setObjectName("report_gen_btn")
        self.report_gen_btn.setToolTip("Generate & View Report")

        
        self.label_3.raise_()
        self.label_4.raise_()
        self.label_7.raise_()
        self.label_9.raise_()
        self.line.raise_()
        self.line_4.raise_()
        self.line_5.raise_()
        self.db_passNo.raise_()
        self.db_name.raise_()
        self.db_gender.raise_()
        self.db_nationality.raise_()
        self.groupBox_4.raise_()
        self.groupBox_5.raise_()
        self.projlabel.raise_()
        self.progressBar.raise_()
        self.groupBox.raise_()
        self.logo.raise_()
        self.db_face.raise_()
        self.person_face.raise_()
        self.passport_face.raise_()
        self.pass_percentage.raise_()
        self.pers_percentage.raise_()
        self.s_status.raise_()
        self.groupBox_3.raise_()
        self.groupBox_2.raise_()
        self.groupBox_6.raise_()
        self.close_btn.raise_()
        self.min_btn.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Forensic Face Matching at Airort during Passport Control"))
        self.projlabel.setText(_translate("MainWindow", "   Forensic Face Matching at Airport During Passport Control"))
        self.close_btn.setText(_translate("MainWindow", ""))
        self.min_btn.setText(_translate("MainWindow", ""))
        self.groupBox.setTitle(_translate("MainWindow", "Passport Verification"))
        self.up_passport_btn.setText(_translate("MainWindow", "Select Passport"))
        self.verify_passport_btn.setText(_translate("MainWindow", "Verify"))
        self.groupBox_2.setTitle(_translate("MainWindow", "MANUAL Passport No. Extraction"))
        self.label_2.setText(_translate("MainWindow", "Enter Passport Number :"))
        self.up_pass_no_btn.setText(_translate("MainWindow", "Enter"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Person Verification"))
        self.up_person_btn.setText(_translate("MainWindow", "Select Person's Image"))
        self.open_cam_btn.setText(_translate("MainWindow", "Camera"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Faces"))
        self.label_5.setText(_translate("MainWindow", "Database Face"))
        self.label_6.setText(_translate("MainWindow", "Passport Face"))
        self.label_8.setText(_translate("MainWindow", "Live Face"))
        self.progressBar.setFormat(_translate("MainWindow", "%p%"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Messege Box"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Passenger\'s Data"))
        self.label_3.setText(_translate("MainWindow", " Name"))
        self.label_4.setText(_translate("MainWindow", " Passport #"))
        self.label_7.setText(_translate("MainWindow", " Gender"))
        self.label_9.setText(_translate("MainWindow", " Nationality"))
        self.report_gen_btn.setText(_translate("Main Window", "View Report"))


        self.var = ''
        self.path_pas = ''
        self.path_pe = ''
        self.count = 5
        self.up_passport_btn.clicked.connect(self.upload_passport)  # Modify
        self.verify_passport_btn.clicked.connect(self.seq_for_pass_verfication)  # Modify
        self.up_pass_no_btn.clicked.connect(self.seq_for_m_pass_verfication)
        self.close_btn.clicked.connect(self.close_win)
        self.min_btn.clicked.connect(self.min_win)
        self.up_person_btn.clicked.connect(self.upload_person)
        self.up_person_btn.setEnabled(False)
        self.logic = False
        self.per_ver = False
        self.value = 1
        self.start_timer = 0
        self.end_timer = 0
        self.open_cam_btn.clicked.connect(self.open_webcam)
        self.open_cam_btn.setEnabled(False)
        self.report_gen_btn.clicked.connect(self.report_gen)
        self.metadata = {"passport": "",
                         "Name": "",
                         "Gender": "", 
                         "Country": "",
                        }
        self.report = {"passport_num": "",
                       "name": "",
                       "gender": "",
                       "country":"",
                       "pn_manual": 0,
                       "passport_face_detected": 0,
                       "passport_verified": 0,
                       "passport_similarity": 0.0,
                       "passport_verification_time": 0.0,
                       "person_face_detected": 0,
                       "person_verified": 0,
                       "person_varification_time": 0.0,
                       "person_similarity": 0.0,
                       "total_time": 0.0,
                       "date": "",
                       "time": ""}
        
    # Interface Maipulation
    def close_win(self):
        try:
            conn.commit()
            cur.close()
            conn.close()

            if os.path.exists('temp/pass_face.png'):
                os.remove('temp/pass_face.png')
                print('Image pass_face.png has been removed on closing')
            if os.path.exists('temp/db_face.png'):
                os.remove('temp/db_face.png')
                print('Image db_face.png has been removed on closing')
            if os.path.exists('temp/person_face.png'):
                os.remove('temp/person_face.png')
                print('Image person_face.png has been removed on closing')
        except:
            print("Error in closing Database and removing pictures")
            sys.exit(503)
        finally:
            sys.exit(0)

    def min_win(self):
        MainWindow.setWindowState(self.windowState() | QtCore.Qt.WindowState.WindowMinimized)

    def cam_per_btn(self, action = False):
        self.up_person_btn.setEnabled(action)
        self.open_cam_btn.setEnabled(action)

    def wind_reset(self):
        self.s_status.setText("")
        self.s_status.setStyleSheet("QLabel{color:'red'; background-color:rgb(245,245,245); font-weight: bold; border: 2px solid gray;}")
        self.db_face.setPixmap(QPixmap(None))
        self.passport_face.setPixmap(QPixmap(None))
        self.person_face.setPixmap(QPixmap(None))
        self.db_gender.setText("")
        self.db_name.setText("")
        self.db_nationality.setText("")
        self.db_passNo.setText("")
        self.person.setPixmap(QPixmap(None))
        self.label_6.setText("Passport Face")
        self.label_8.setText("Live Face")
        self.progressBar.setValue(0)
        self.progressBar.setProperty('value', self.progressBar.value())
        self.pass_no_le.setText("")
        self.pass_box(message="", color="rgb(220, 230, 220)")
        self.pers_box(message="", color="rgb(220, 230, 220)")

    def prog_update(self, value, inc):
        if self.progressBar.value() != value:
            self.progressBar.setValue(value)
        if self.progressBar.value() == value:
            self.progressBar.setValue(self.progressBar.value() + inc)
            self.progressBar.setProperty('value', self.progressBar.value())

    def all_btn(self, action=False):
        self.open_cam_btn.setEnabled(action)
        self.up_person_btn.setEnabled(action)
        self.up_passport_btn.setEnabled(action)
        self.verify_passport_btn.setEnabled(action)
        self.up_pass_no_btn.setEnabled(action)
        self.report_gen_btn.setEnabled(action)

    def pers_box(self, message="", color=""):
        self.pers_percentage.setText(message)
        self.pers_percentage.setStyleSheet("QLabel{"+"background-color: {};".format(color)+"border: 1px solid gray}")

    def pass_box(self, message="", color=""):
        self.pass_percentage.setText(message)
        self.pass_percentage.setStyleSheet("QLabel{"+"background-color: {};".format(color)+"border: 1px solid gray}")

    def status_update(self, message="", color='', bg_color="rgb(245, 245, 245)"):
        self.s_status.setText(message)
        self.s_status.setStyleSheet(("QLabel{" + "color:'{}';".format(color) + "background-color:{};".format(bg_color)+" font-weight: bold; border: 2px solid gray}"))

    # Functionalities & Modules
    def upload_passport(self):
        local_log = []
        try:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            if os.path.exists("temp/db_face.png"):
                os.remove("temp/db_face.png")
            if os.path.exists("temp/pass_face.png"):
                os.remove("temp/pass_face.png")
            self.report = {"passport_num": "",
                       "name": "",
                       "gender": "",
                       "country": "",
                       "pn_manual": 0,
                       "passport_face_detected": 0,
                       "passport_verified": 0,
                       "passport_similarity": 0,
                       "passport_verification_time": 0,
                       "person_face_detected": 0,
                       "person_verified": 0,
                       "person_varification_time": 0,
                       "person_similarity": 0,
                       "total_time": 0,
                       "date": "",
                       "time": ""}
            
            self.start_timer = time.time()
            self.cam_per_btn(action=False)
            self.wind_reset()
            filename = QFileDialog.getOpenFileName()
            filename = filename[0]
            if(len(filename) != 0):
                if filetype.is_image(filename):
                    self.var = "-- Selected file is valid image.\n" + self.var
                    self.path_pas = filename
                    self.passport.setPixmap(QPixmap(self.path_pas))
                    self.status_update(message="", color="black")
                    self.var = "-- Passport Data Page is uploaded successfully.\n   Path: " + self.path_pas  +"\n"+ self.var
                    self.plainTextEdit.setPlainText(self.var)
                    self.prog_update(value=0, inc=10)
                    message = "Passport Data Page is uploaded successfully.\nPath: " + self.path_pas
                else:
                    self.status_update(message="Select the Passport Image!", color="red")
                    self.var = "-- Selected file is NOT an Image.\n" + self.var
                    self.plainTextEdit.setPlainText(self.var)
                    message = "Passport Data Page is not selected"
            else:
                self.path_pas = ''
                self.status_update(message="Please Upload Passport Image!", color="red")
                self.passport.setPixmap(QPixmap(None))
                self.plainTextEdit.setPlainText(self.var)
                message = "Passport Data page is not uploded"
        except Exception as e:
            self.var = "-- Error Occur: " + str(e) + "\n" + self.var
            message = "-- Error Occur: " + str(e)
            self.plainTextEdit.setPlainText(self.var)

        self.log_reg(log_hist=message, passport_no="")

    def seq_for_pass_verfication(self):
        if (len(self.path_pas) != 0):
            curr_time = 0
            pn = self.passport_num_extract()
            if (pn != 0):
                dbf = self.fetch_DB_photo(pn)
                if (dbf != 0):
                    self.db_name.setText(" "+self.metadata["Name"])
                    if self.metadata["Gender"] == 'M':
                        self.db_gender.setText(" Male")
                    elif self.metadata["Gender"] == 'F':
                        self.db_gender.setText(" Female")
                    self.db_passNo.setText(str(" "+self.metadata['passport']))
                    self.db_nationality.setText(" "+self.metadata["Country"])
                    start = time.time()
                    pf = self.pass_face_detect()
                    if (pf != 0):
                        self.pass_verification()
                        end = time.time()
                        curr_time = end - start
                    self.cam_per_btn(action=True)
                    self.report["passport_verification_time"] = curr_time
            self.plainTextEdit.setPlainText(self.var)
        else:
            self.status_update(message="Please Upload the Passport Image!", color="red")
            self.plainTextEdit.setPlainText(self.var)
            message = "Passport Image not uploaded"
            self.log_reg(log_hist=message, passport_no="")

    def seq_for_m_pass_verfication(self):
        if (len(self.path_pas) != 0):
            curr_time = 0
            pn = self.pass_no_le.text()
            if (len(pn) != 9):
                self.status_update("Invalid Passport Number! 🚫", color="red")
                message = "Invalid Passport Number. Passport Number Entered: {}".format(pn)
                self.log_reg(log_hist=message, passport_no=pn)
            else:
                if os.path.exists("temp/db_face.png"):
                    os.remove("temp/db_face.png")
                if os.path.exists("temp/pass_face.png"):
                    os.remove("temp/pass_face.png")
                
                self.cam_per_btn(action=False)
                self.wind_reset()
                # self.passport.setPixmap(QPixmap(self.path_pas))
                self.report['pn_manual'] = 1
                self.report['passport_num'] = pn
                self.prog_update(value=10, inc=10)
                dbf = self.fetch_DB_photo(pn)
                if (dbf != 0):
                    self.db_name.setText(" "+self.metadata["Name"])
                    if self.metadata["Gender"] == 'M':
                        self.db_gender.setText(" Male")
                    elif self.metadata["Gender"] == 'F':
                        self.db_gender.setText(" Female")
                    self.db_passNo.setText(str(" "+self.metadata['passport']))
                    self.db_nationality.setText(" "+self.metadata["Country"])
                    start = time.time()
                    pf = self.pass_face_detect()
                    if (pf != 0):
                        self.pass_verification()
                        end = time.time()
                        curr_time = end - start
                    self.report["passport_verification_time"] = curr_time
                    self.cam_per_btn(action=True)
            self.plainTextEdit.setPlainText(self.var)
        else:
            self.status_update("Please Upload the Passport Image!", color="red")
            self.passport.setPixmap(QPixmap(None))
            self.plainTextEdit.setPlainText(self.var)
            message = "Passport Image not uploaded"
            self.log_reg(log_hist=message, passport_no="")

    def passport_num_extract(self):
        mrz = read_mrz(self.path_pas)
        if(mrz == None):
            self.status_update(message="Passport number extraction FAILED ❌.", color="red")
            message = "Passport Number Extraction failed"
            val = 0
        else:
            mrz_data = mrz.to_dict()
            pn = mrz_data['number']
            self.metadata["passport"] = pn
            self.report['passport_num'] = pn
            self.var = "-- Passport Number is extracted successfully.\n   Passport Number:"+ pn +"\n"+ self.var
            message = "Passport Number Extraction Completed. Passport Number: {}".format(pn)
            val = self.metadata['passport']
        self.prog_update(value=10, inc=10)
        self.log_reg(log_hist=message, passport_no=str(val))
        return val

    def fetch_DB_photo(self, pn):
        pass_no = pn
        cur.execute('SELECT name, gender, country, photo FROM Users WHERE pass_no = ?', (pass_no,))
        db_data = cur.fetchall()
        val = 0
        if (len(db_data) == 0):
            self.status_update(message="Passport Number NOT found in Database! ❌", color="red")
            message = "Passport Number not Found in Database"
        else:
            for i in db_data:
                person_name = i[0]
                self.metadata["passport"] = pass_no
                self.metadata["Name"] = person_name
                self.report['name'] = person_name
                self.report["gender"] = i[1]
                self.report["country"] = i[2]
                self.metadata["Gender"] = i[1]
                self.metadata["Country"] = i[2]
                person_photo = i[3]
            db_f = "temp/db_face.png"
            with open(db_f, 'wb') as f:
                f.write(person_photo)
            f.close()

            temp = face_process(db_f, plot=False)
            temp.save(db_f)
            temp.close()
            self.db_face.setPixmap(QPixmap(db_f))
            self.var = "-- Database Photo Fetched successfully.\n   Person Name: " + person_name + "\n" +"-- Passport Number Matched successfully.\n" +  self.var
            message = "Database photo fetched successfully. Person Name:{}\nPassport Number Matched Successfully".format(person_name)
            val = 1
        self.prog_update(value=20, inc=10)
        self.log_reg(log_hist=message, passport_no=str(self.metadata['passport']))
        return val
        
    def pass_face_detect(self):
        path = self.path_pas
        face_location = face_process(path, plot=False)
        val = 0
        if (face_location is None):
            self.status_update(message="Face NOT detected! ❌", color="red", bg_color='rgb(255, 215, 215)')
            message = "Face not detected in the photo"
        else:
            pass_f = "temp/pass_face.png"
            face_location.save(pass_f)
            self.passport_face.setPixmap(QPixmap(pass_f))
            face_location.close()
            message = "Face detected successfully"
            val = 1
        self.prog_update(value=30, inc=10)
        self.report["passport_face_detected"] = val
        self.log_reg(log_hist=message, passport_no=str(self.metadata['passport']))
        return val

    def pass_verification(self):
        results = self.ver_module('pass_face.png', 'db_face.png')
        result = round(results * 100)
        if results >= 0.75:
            print(results)
            verify = 1
            self.pass_box(message="{}%".format(result), color="rgb(215, 255, 215)")
            self.status_update(message="Passport is verified ✔️", color='green', bg_color='rgb(215, 255, 215)')
            self.label_6.setText("Passport face ✔️")
            message = "Face on passport matched successfully\n"
            if self.count <= 0:
                self.count = 5
            
        else:
            print(results)
            verify = 0
            self.status_update(message="Passport Face Not Matched! ❌", color='red', bg_color='rgb(255, 215, 215)')
            self.label_6.setText("Passport Face ❌")
            
            self.pass_box(message="{}%".format(result), color="rgb(255, 215, 215)")
            message = "Face on passport NOT matched"
            self.count = 5
        self.prog_update(value=40, inc=10)
        self.report["passport_verified"] = verify
        self.report["passport_similarity"] = float(results)
        self.log_reg(log_hist=message, passport_no=str(self.metadata['passport']))


    def upload_person(self):
        if os.path.exists("temp/person_face.png"):
            os.remove("temp/person_face.png")
        self.person_face.setPixmap(QPixmap(None))
        self.person.setPixmap(QPixmap(None))
        self.pers_box(message="", color="rgb(220, 230, 220)")
        filename = QFileDialog.getOpenFileName()
        filename = filename[0]
        if (len(filename) != 0):
            if filetype.is_image(filename):
                self.var = "-- Selected file is valid image.\n" + self.var
                self.path_pe = filename
                self.person.setPixmap(QPixmap(self.path_pe))
                self.prog_update(value=50, inc=10)
                self.status_update(message="", color='black')
                self.var = "-- (Person's Photo is uploaded successfully.)\nPath: " + self.path_pe + "\n" + self.var
                self.plainTextEdit.setPlainText(self.var)
                message = "Selected image is a valid image. Path: {}".format(self.path_pe)
                self.seq_for_person_verfication()
            else:
                self.s_status.setText("Select valid image!")
                self.s_status.setStyleSheet("QLabel{color:'red'; background-color:rgb(255,235,235); font-weight: bold; border: 2px solid gray;}")
                self.var = "-- Selected file is NOT an Image.\n" + self.var
                message = "Selected Invalid Image. Path: {}".format(filename)
                self.plainTextEdit.setPlainText(self.var)
        else:
            self.s_status.setText("Please select Person's Image!")
            self.s_status.setStyleSheet("QLabel{color:'red'; background-color:rgb(255,235,235); font-weight: bold; border: 2px solid gray;}")
            self.plainTextEdit.setPlainText(self.var)
            self.label_8.setText("Live Face")
            message = "Image is not selected"
        
        self.log_reg(log_hist=message, passport_no=str(self.metadata['passport']))

    def seq_for_person_verfication(self):
        if (len(self.path_pe) != 0):
            curr_time = 0
            if self.count <= 5 and self.count > 0:
                self.progressBar.setValue(60)
                self.progressBar.setProperty('value', self.progressBar.value())
                start = time.time()
                pf = self.per_face_detect()
                if (pf != 0):
                    self.per_verification()
                    end = time.time()
                    if self.per_ver is True:
                        self.count = 5
                    elif self.per_ver is False:
                        self.count = self.count  - 1
                    else:
                        print('Try again...')
                    curr_time = end - start
                self.report["person_varification_time"] = curr_time
                self.report_write()
            else: 
                self.status_update(message="⚠️ Verification attempts limit Exceeded", color='red')
                self.plainTextEdit.setPlainText('-- Person Verification attempt limits exceeded.\n> Person does not have any match with the Database.\n> Call the Authorities!\n')
                message = "Verification attempts limit Exceeded"
                self.pass_box(message="", color="rgb(220, 230, 220)")
                self.label_8.setText("Live Face")
                self.log_reg(log_hist=message, passport_no=self.metadata['passport'])
        else:
            self.status_update(message="Please select Person's Image!", color="red")
            message = "Image is not selected"
            self.label_8.setText("Live Face")
            self.plainTextEdit.setPlainText(self.var)
            self.log_reg(log_hist=message, passport_no=str(self.metadata['passport']))

    def per_face_detect(self):
        path = self.path_pe
        val = 0
        person_img = face_process(path, plot=False)
        if person_img is None:
            self.status_update(message="Person Face NOT Detected", color='red', bg_color='rgb(255, 215, 215)')
            message = "Person face NOT Detected"
        else:
            per_f = "temp/person_face.png"
            person_img.save(per_f)
            
            self.person_face.setPixmap(QPixmap(per_f))
            message  = "Person face detected"
            val = 1
        self.prog_update(value=60, inc=20)
        self.report["person_face_detected"] = val
        self.log_reg(log_hist=message, passport_no=self.metadata['passport'])
        return val

    def per_verification(self):
        if (len(self.path_pas) != 0):
            results = self.ver_module('db_face.png', 'person_face.png')
            result = round(results * 100)
            verify = 0
            if results >= 0.5:
                print(results)
                verify = 1
                
                self.status_update(message="Person is verified   ✔️", color='green', bg_color='rgb(215, 255, 215)')
                self.label_8.setText("Live Face ✔️")
                self.pers_box(message="{}%".format(result), color="rgb(215,255,215)")
                message = "Person is verified"
                self.per_ver = True
                
            else:
                print(results)
                self.status_update(message="Person is not verified   ❌", color='red', bg_color='rgb(255, 215, 215)')
                self.label_8.setText("Live Face ❌")
                self.pers_box(message="{}%".format(result), color="rgb(255, 215, 215)")
                self.per_ver = False
                message = "Person is not verified"
            self.prog_update(value=80, inc=20)
            self.report["person_verified"] = verify
            self.report["person_similarity"] = float(results)
        else:
            self.status_update(message="⚠️ Please Upload the Passport Image!", color='red')
            message = 'Passport Image has not uploaded'
        self.end_timer = time.time()
        self.report["total_time"] = self.end_timer - self.start_timer
        self.log_reg(log_hist=message, passport_no=self.metadata['passport'])
    
    def open_webcam(self):
        self.status_update(message="Opening Live Video Feed...", color='black')
        cap_f = "temp/person_face.png"
        self.all_btn(action=False)
        gaze_direction = "center"
        if os.path.exists(cap_f):
            os.remove(cap_f)
        try:
            cap = cv2.VideoCapture(0)
            self.log_reg(log_hist='Camera Opens', passport_no=self.metadata['passport'])
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                faces_dlib = face_detector(frame)
                roi_color = frame.copy()
                self.status_update(message="🔍 Detecting Face...", color='red')
                for (x, y, w, h), face in zip(faces, faces_dlib):
                    color = (0, 0, 255) # Default Red
                    landmarks = landmark_detector(frame, face)
                    points = [(p.x, p.y) for p in landmarks.parts()]
                    left_eye = landmarks.part(36)
                    right_eye = landmarks.part(45)
                    nose = landmarks.part(30)

                    left_eye_dist = int(np.round(np.sqrt((nose.x - left_eye.x)**2 + (nose.y - left_eye.y)**2)))
                    right_eye_dist = int(np.round(np.sqrt((nose.x - right_eye.x)**2 + (nose.y - right_eye.y)**2)))

                    if x + w//2 > gray.shape[1]//2 - w//4 and x + w//2 < gray.shape[1]//2 + w//4:
                        color = (0, 255, 0) # Turns green if face is detected
                        if left_eye_dist > right_eye_dist:
                            gaze_direction = "right"
                            self.status_update(message="Turn your Face LEFT...", color='green')
                        elif left_eye_dist < right_eye_dist:
                            gaze_direction = "left"
                            self.status_update(message="Turn your Face RIGHT...", color='green')
                        elif left_eye_dist == right_eye_dist:
                            gaze_direction = "center"
                            

                        print("Right Eye:   ", right_eye_dist, "    Left Eye:   ", left_eye_dist)


                        if gaze_direction == "center":
                            self.status_update(message="Face Detected! Taking Picture... ✔️", color='green')
                            self.log_reg(log_hist='Face captured', passport_no=self.metadata['passport'])
                            print("Face Captured...")
                            if self.logic is False:
                                self.logic = True
                        
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)

                self.displayImage(frame)

                cv2.waitKey(1)

                if self.logic is True:
                    print('Taking Picture...')
                    self.log_reg(log_hist='Picture taken', passport_no=self.metadata['passport'])
                    cv2.imwrite(cap_f, roi_color)
                    self.logic = False
                    self.person.setPixmap(QPixmap(cap_f))
                    self.path_pe = cap_f
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.prog_update(value=50, inc=10)
            self.seq_for_person_verfication()
        except Exception as e:
            self.status_update(message="Problem with the CAMERA! Kindly Check...", color='red')
            print("Error: {}".format(e))
           
        self.all_btn(action=True)
        

    def displayImage(self, img):
        qformat = QImage.Format.Format_Indexed8
        # print(qformat)
        if len(img.shape) == 3:
            if(img.shape[2]) == 4:
                qformat = QImage.Format.Format_RGBA8888
            else:
                qformat = QImage.Format.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.person.setPixmap(QPixmap.fromImage(img))

    def ver_module(self, main_face_to_compare_with='',  face_to_compare=''):
        root = 'temp/'
        images = []
        for fname in [main_face_to_compare_with, face_to_compare]:
            path = os.path.join(root, fname)
            input_img = Image.open(path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ])
            input_img = transform(input_img).unsqueeze(0)
            images.append(input_img)
                
        images = torch.cat(images, dim=0)

        x_vec = mtlface.encode((images))
        results = dot(x_vec[0], x_vec[1]) / (norm(x_vec[0]) * norm(x_vec[1]))

        return results

    def log_reg(self, log_hist="",  passport_no=""):
        name_tuple = time.localtime()
        localDatetime = time.strftime("%m-%d-%Y, %H:%M:%S", name_tuple)
        
        local = []
        local.append(localDatetime)
        pn = passport_no
        local.append(pn)
        local.append(log_hist)
        
        log_file_path = 'log/immigrationLogs.csv'
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w+") as f:
                pass
            f.close()

        columns = ['Datetime', 'Passport #', 'Log History']
        df_log = pd.read_csv(log_file_path, names=columns, header=0)

        local[0] = localDatetime
        local[1] = pn
        local[2] = log_hist
        dataframe = []
        dataframe.append(local)
        dataframe = pd.DataFrame(dataframe, columns=columns)
        df_log = pd.concat([df_log, dataframe])

        df_log.to_csv(log_file_path, index=False, columns=columns, header=True)

    def report_write(self):
        try:
            conn = sqlite3.connect('Database/immigration_DB.db')
            cur = conn.cursor()

            name_tuple = time.localtime()
            self.report['date'] = time.strftime("%m-%d-%Y", name_tuple)
            self.report['time'] = time.strftime("%H:%M:%S", name_tuple)
            cur.execute("""INSERT INTO Report (passport_num, 
                            name,
                            gender,
                            country, 
                            pn_manual, 
                            passport_face_detected, 
                            passport_verified, 
                            passport_similarity, 
                            passport_verification_time, 
                            person_face_detected,                                                                                                 
                            person_similarity, 
                            person_verified, 
                            person_varification_time, 
                            total_time, 
                            date, 
                            time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""", (self.report['passport_num'], self.report['name'], self.report['gender'],
                                                                            self.report['country'], self.report['pn_manual'], 
                                                                            self.report['passport_face_detected'], self.report['passport_verified'],
                                                                            self.report['passport_similarity'],
                                                                            self.report["passport_verification_time"], self.report['person_face_detected'],
                                                                            self.report['person_similarity'], self.report["person_verified"],
                                                                            self.report['person_varification_time'], self.report["total_time"], 
                                                                            self.report['date'], self.report['time']))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("Failure with database or entering in data")
            print("Erro: "+ str(e))
            sys.exit(503)


    def report_gen(self):
        def dlg_show(message="", icon=QMessageBox.Icon.Information, on_console=""):
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Report Generation")
            pixmapi = QStyle.StandardPixmap.SP_TitleBarMenuButton
            win_icon = self.style().standardIcon(pixmapi)
            dlg.setWindowIcon(win_icon)
            dlg.setText(message)
            dlg.setIcon(icon)
            button = dlg.exec()
            if button == QMessageBox.StandardButton.Ok:
                print(on_console)
            
        cwd = os.getcwd()           
        try:
            name_tuple = time.localtime()
            root_folder = "Report"
            if not os.path.exists(root_folder):
                os.mkdir(root_folder)
            date = time.strftime("%m-%d-%Y %H.%M.%S", name_tuple)
            pdf_filename = "report {}.pdf".format(date)
            folder = root_folder + "/Report {}".format(date)
            if not os.path.exists(folder):
                os.mkdir(folder)
            generate_report(filename=folder+'/'+pdf_filename)

            conn = sqlite3.connect('Database/immigration_DB.db')
            columns = ["passport_num", 
                            "name",
                            "gender",
                            "country", 
                            "pn_manual",
                            "passport_face_detected", 
                            "passport_verified", 
                            "passport_similarity", 
                            "passport_verification_time", 
                            "person_face_detected", 
                            "person_similarity", 
                            "person_verified", 
                            "person_varification_time", 
                            "total_time", 
                            "date", 
                            "time"]
            

            csv_filename = "report {}.csv".format(date)
            sql_table = pd.read_sql_query("SELECT * FROM 'Report'", con=conn)
            df = pd.DataFrame(sql_table, columns=columns)
            df.to_csv(folder+'/'+csv_filename, index=False)

            pixmapi = QStyle.StandardPixmap.SP_MessageBoxInformation
            icon = self.style().standardIcon(pixmapi)
            message = "Report generated Successfully.\n\nFor more details, visit: \"{}\"".format(os.path.join(cwd, folder))
            dlg_show(message=message, icon=QMessageBox.Icon.Information, on_console="Report Generated Successfully!")
            webbrowser.open_new(os.path.join(cwd, folder+'/'+pdf_filename))

        except Exception as e:
            print("Error: {}".format(e))
            message = "Report Generation failed.\n\nError: {}".format(e)
            pixmapi = QStyle.StandardPixmap.SP_MessageBoxInformation
            icon = self.style().standardIcon(pixmapi)
            dlg_show(message=message, icon=QMessageBox.Icon.Critical, on_console="Report Generated Failed!")


# Driver Code
if __name__ == "__main__":
    try:
        conn = sqlite3.connect('Database/immigration_DB.db')
        cur = conn.cursor()
    except Exception as e:
        print("Database connection failed")
        print("Error: {}".format(e))
        sys.exit(503)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow=MainWindow)
    MainWindow.show()
    
    try:
        mtlface = torch.load('Resources/model/Smart_One.pth', map_location=torch.device('cpu'))
        face_cascade = cv2.CascadeClassifier('Resources/detectors/haarcascade_frontalface_default.xml')
        face_detector = dlib.get_frontal_face_detector()
        landmark_detector = dlib.shape_predictor("Resources/detectors/shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print("Problem in loading Model(s)! Kindly check if the model is present in the path \"/Resources/model/\""
                +"\nOr there Might be a problem while loading the model...")
        print("Error: {}".format(e))
        sys.exit(404)

    app.exec()