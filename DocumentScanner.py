# Hector Moya
# Mechanical CAD
# Final Project - Document Scanner

################################################################################
# =========================== Python Version 3.8 ============================= #
################################################################################

# This program will ask for a picture file of a document, perform several image processing techniques, and
# provide a "document scan" or top down view of the image and save it in the same file directory

# *********************************************************************************************************************
# *--------------------------------------------- Packages ------------------------------------------------------------*
# *********************************************************************************************************************
# Import the necessary packages
from tkinter import *  # Standard Python interfaces to the Tk GUI toolkit

from skimage.filters import threshold_local  # A collection of algorithms for image processing
import cv2  # OpenCV is an open source computer vision and machine learning software library
import imutils  # A series of convenience functions to make basic image processing functions
from tkinter import filedialog  # Contains convenience classes and functions for creating simple modal dialogs
# to get a value from the user
import numpy as np  # Fundamental package for scientific computing
import os  # Provides a portable way of using operating system dependent functionality

# ---------------------------------------------------------------------------
# These imports are required to make the GUI for the application work
import sys
from PySide2 import QtCore
from PySide2.QtCore import QRect, QSize, Qt
from PySide2.QtGui import QColor, QFont, QIcon
from PySide2.QtWidgets import QFrame, QLabel, QHBoxLayout, QApplication, \
    QPushButton, QVBoxLayout, QWidget, \
    QGraphicsDropShadowEffect, QMainWindow


# ---------------------------------------------------------------------------


# *********************************************************************************************************************
# *-------------------------------------------- Functions ------------------------------------------------------------*
# *********************************************************************************************************************
# This function takes a single argument, pts , which is a list of four points
# specifying the coordinates of each point of the rectangle
def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates
    return rect


# This function requires two arguments: image and pts. It determines the dimensions of the newly warped image
# and attributes them to 4 points. These points are defined in a consistent ordering representation
# and allow us to view a top-down view of the image
def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Now we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # Return the warped image
    return warped


# This function will do the rest of the workload. It will apply several image processing techniques
# that will first find the edges of the document, scan the contours and transform the image to a more
# recognizable form that is saved in the same file path as the original photo under a different name
def docuScan():
    # Load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    global screenCnt
    filepath = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpeg"), ("jpg files", "*.jpg"),
                                                     ("png files", "*.png"), ("all files", "*.*")))
    filename = os.path.basename(filepath)
    image = cv2.imread(filepath)
    ratio = image.shape[0] / 800.0
    orig = image.copy()
    image = imutils.resize(image, height=800)
    # Convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Find the contours in the edged image, keeping only
    # the largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # Loop over the contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # If our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

    # Apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # Convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    # Show the scanned images
    cv2.imshow("Scanned", imutils.resize(warped, height=800))
    cv2.waitKey(0)
    newFilename = filepath.split(".")[0] + "_scanned" + "." + filename.split(".")[1]
    cv2.imwrite(newFilename, imutils.resize(warped, height=800))


# *********************************************************************************************************************
# *------------------------------------------------ GUI Window -------------------------------------------------------*
# *********************************************************************************************************************

# This will be used to bring up the file window dialogue
window = Tk()  # Initialize the window
window.withdraw()  # This is to now show the empty Tk Window when bringing up the file directory dialog


# This function will be used by the "Open File to Scan" button do run the DocuScan() function
def openfilePrompt():
    docuScan()


# This class will be used to define our window elements and effects
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.openfile_button.clicked.connect(openfilePrompt)
        self.setWindowIcon(QIcon("docscanicon.ico"))

        # Move window
        def moveWindow(event):
            # If left click move window
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # Set title bar
        self.ui.title_bar.mouseMoveEvent = moveWindow
        # Set UI definitions
        UIFunctions.uiDefinitions(self)
        # Show main window
        self.show()

    # Tracks the position of the mouse if pressed
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


# Globals
GLOBAL_STATE = 0


# This class will be used to set all the UI functions
class UIFunctions(MainWindow):
    # ==> UI DEFINITIONS
    def uiDefinitions(self):
        # REMOVE TITLE BAR
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # SET DROP SHADOW WINDOW
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 100))
        # APPLY DROP SHADOW TO FRAME
        self.ui.drop_shadow_frame.setGraphicsEffect(self.shadow)
        # MINIMIZE
        self.ui.button_minimize.clicked.connect(lambda: self.showMinimized())
        # CLOSE
        self.ui.button_close.clicked.connect(lambda: self.close())

    # RETURN STATUS IF WINDOWS IS MAXIMIZE OR RESTORED
    def returnStatus(self):
        return GLOBAL_STATE


# This piece of code is imported in from the Qt Designer interface where the application UI is created
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(250, 150)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.drop_shadow_layout = QVBoxLayout(self.centralwidget)
        self.drop_shadow_layout.setContentsMargins(10, 10, 10, 10)
        self.drop_shadow_layout.setSpacing(0)
        self.drop_shadow_layout.setObjectName("drop_shadow_layout")
        self.drop_shadow_frame = QFrame(self.centralwidget)
        self.drop_shadow_frame.setStyleSheet(
            "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(42, 44, 111, 255), "
            "stop:0.479212 rgba(28, 29, 73, 255));\n"
            "border-radius: 10px;")
        self.drop_shadow_frame.setFrameShape(QFrame.NoFrame)
        self.drop_shadow_frame.setFrameShadow(QFrame.Raised)
        self.drop_shadow_frame.setObjectName("drop_shadow_frame")
        self.verticalLayout = QVBoxLayout(self.drop_shadow_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.title_bar = QFrame(self.drop_shadow_frame)
        self.title_bar.setFrameShape(QFrame.StyledPanel)
        self.title_bar.setFrameShadow(QFrame.Raised)
        self.title_bar.setObjectName("title_bar")
        self.horizontalLayout = QHBoxLayout(self.title_bar)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_title = QFrame(self.title_bar)
        self.frame_title.setStyleSheet("background: none;")
        self.frame_title.setFrameShape(QFrame.StyledPanel)
        self.frame_title.setFrameShadow(QFrame.Raised)
        self.frame_title.setObjectName("frame_title")
        self.label = QLabel(self.frame_title)
        self.label.setGeometry(QRect(0, 0, 111, 21))
        font = QFont()
        font.setFamily("Roboto")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 170, 0);")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.frame_title)
        self.frame_buttons = QFrame(self.title_bar)
        self.frame_buttons.setStyleSheet("background: none;")
        self.frame_buttons.setFrameShape(QFrame.StyledPanel)
        self.frame_buttons.setFrameShadow(QFrame.Raised)
        self.frame_buttons.setObjectName("frame_buttons")
        self.button_minimize = QPushButton(self.frame_buttons)
        self.button_minimize.setGeometry(QRect(60, 0, 16, 16))
        self.button_minimize.setMinimumSize(QSize(16, 16))
        self.button_minimize.setMaximumSize(QSize(16, 16))
        self.button_minimize.setStyleSheet("QPushButton {\n"
                                           "    border: none;\n"
                                           "    border-radius: 8px;\n"
                                           "    background-color: rgb(255, 170, 0);\n"
                                           "}\n"
                                           "QPushButton:hover {\n"
                                           "    \n"
                                           "    background-color: rgba(255, 170, 0, 150);\n"
                                           "}")
        self.button_minimize.setText("")
        self.button_minimize.setObjectName("button_minimize")
        self.button_close = QPushButton(self.frame_buttons)
        self.button_close.setGeometry(QRect(90, 0, 16, 16))
        self.button_close.setMinimumSize(QSize(16, 16))
        self.button_close.setMaximumSize(QSize(16, 16))
        self.button_close.setStyleSheet("QPushButton {\n"
                                        "    border: none;\n"
                                        "    border-radius: 8px;\n"
                                        "    \n"
                                        "    background-color: rgb(255, 0, 0);\n"
                                        "}\n"
                                        "QPushButton:hover {\n"
                                        "    \n"
                                        "    background-color: rgba(255, 0, 0, 150);\n"
                                        "}")
        self.button_close.setText("")
        self.button_close.setObjectName("button_close")
        self.horizontalLayout.addWidget(self.frame_buttons)
        self.verticalLayout.addWidget(self.title_bar)
        self.content_bar = QFrame(self.drop_shadow_frame)
        self.content_bar.setFrameShape(QFrame.StyledPanel)
        self.content_bar.setFrameShadow(QFrame.Raised)
        self.content_bar.setObjectName("content_bar")
        self.horizontalLayout_3 = QHBoxLayout(self.content_bar)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.openfile_button = QPushButton(self.content_bar)
        self.openfile_button.setMinimumSize(QSize(115, 25))
        self.openfile_button.setMaximumSize(QSize(115, 25))
        self.openfile_button.setStyleSheet("QPushButton {\n"
                                           "    border: none;\n"
                                           "    border-radius: 8px;\n"
                                           "    \n"
                                           "    \n"
                                           "    background-color: rgb(255, 170, 0);\n"
                                           "}\n"
                                           "QPushButton:hover {\n"
                                           "    \n"
                                           "    background-color: rgb(255, 170, 0, 150);\n"
                                           "}")
        self.openfile_button.setObjectName("openfile_button")
        self.horizontalLayout_3.addWidget(self.openfile_button)
        self.verticalLayout.addWidget(self.content_bar)
        self.credits_bar = QFrame(self.drop_shadow_frame)
        self.credits_bar.setMaximumSize(QSize(16777215, 30))
        self.credits_bar.setFrameShape(QFrame.StyledPanel)
        self.credits_bar.setFrameShadow(QFrame.Raised)
        self.credits_bar.setObjectName("credits_bar")
        self.horizontalLayout_2 = QHBoxLayout(self.credits_bar)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_credits = QLabel(self.credits_bar)
        self.label_credits.setStyleSheet("color: rgb(105, 76, 106)")
        self.label_credits.setObjectName("label_credits")
        self.horizontalLayout_2.addWidget(self.label_credits)
        self.verticalLayout.addWidget(self.credits_bar)
        self.drop_shadow_layout.addWidget(self.drop_shadow_frame)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("MainWindow")
        self.label.setText("DocScanner")
        self.openfile_button.setText("Open File to Scan")
        self.label_credits.setText("By: Hector Moya")


# Initialize the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
