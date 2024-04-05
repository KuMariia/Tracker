import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget
)
from PyQt5.QtCore import (
    Qt,
    QObject,
    QThread,
    pyqtSignal,
    pyqtSlot,
    QPoint,
    QSize
)
from PyQt5.QtGui import QImage, QPixmap
import cv2
import logging
import sys
import os

from tracker import Tracker
tracker = Tracker()
single_tracker = cv2.TrackerMIL.create()

logging.basicConfig(format="%(message)s", level=logging.INFO)
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
BASE_VIDEO_WIDTH = 1280
BASE_VIDEO_HEIGHT = 720

# sys.stdout = open(os.devnull, "w")
# sys.stderr = open(os.path.join(os.getenv("TEMP"), "stderr-"+os.path.basename(sys.argv[0])), "w")

def update_single_tracker(frame):
    success, bbox = single_tracker.update(frame)
    if success:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
    else:
        x1, y1, x2, y2 = -1, -1, -1, -1
    return [x1, y1, x2, y2], success


class Window(QMainWindow):
    recievedFrame = pyqtSignal()
    clickedMouse = pyqtSignal(str, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_count = 0
        self.setupUi()
        self.setupThreads()
    
    def setupUi(self):
        self.setWindowTitle('Single target observation')
        # Each window has one central widget. Here's variable 'centralWidget'
        # I can rename this, it's not reserved name
        self.centralWidget = QWidget()
        # 'setCentralWidget' is a reserved function
        self.setCentralWidget(self.centralWidget)
        # Create 'real', visible widgets: video and button with textBox
        # They gonna be inside a central widget
        # Pay your attention! When you want to STREAM a video,
        # use QLabel - it's used to store text or image,
        # because stream is 'slide show'. Don't use QVideoWidget -
        # it's used inside QMediaPlayer to show recorded video, not a stream
        self.webCamLabel = QLabel()
        self.webCamLabel.resize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.webCamLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # Define function on mouse pressing
        self.webCamLabel.mousePressEvent = self.onWebCamPressed
        
        self.clicksLabel = QLabel('Counting: 0 clicks', self)
        self.clicksLabel.setAlignment(Qt.AlignHCenter)
        
        self.countBtn = QPushButton('Click me!', self)
        # Connect widgets to slots
        self.countBtn.clicked.connect(self.onCountBtnClick)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.webCamLabel)
        layout.addWidget(self.clicksLabel)
        layout.addWidget(self.countBtn)
        self.centralWidget.setLayout(layout)

    def setupThreads(self):
        # Make a thread that shows video. Main thread can't do this,
        # otherwise, GUI would be frozen
        self.wcThread = QThread()
        self.wcWorker = WebCamWorker()

        self.wcWorker.moveToThread(self.wcThread)

        self.wcThread.started.connect(self.wcWorker.init)
        self.wcThread.started.connect(self.wcWorker.run)
        self.wcThread.finished.connect(self.wcWorker.finish_app)
        self.wcThread.finished.connect(self.wcWorker.deleteLater)
        self.wcThread.finished.connect(self.wcThread.deleteLater)

        self.wcWorker.changePixmap.connect(self.setImage)

        self.recievedFrame.connect(self.wcWorker.run)
        self.clickedMouse.connect(self.wcWorker.handle_mouse_click)

        self.wcThread.start()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.webCamLabel.setPixmap(QPixmap.fromImage(image))
        self.recievedFrame.emit()
    
    def onCountBtnClick(self):
        self.click_count += 1
        self.clicksLabel.setText(f'Counting: {self.click_count} clicks')
    
    def onWebCamPressed(self, e):
        self.clickedMouse.emit('LEFT' if e.button() == Qt.LeftButton else 'RIGHT', e.pos())

    def closeEvent(self, e):
        self.wcThread.quit()
        self.wcThread.wait()
        e.accept()

    def resizeVideo(self, size):
        global VIDEO_WIDTH, VIDEO_HEIGHT
        VIDEO_WIDTH = size.width()
        VIDEO_HEIGHT = size.height()
        self.webCamLabel.resize(VIDEO_WIDTH, VIDEO_HEIGHT)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            global BASE_VIDEO_WIDTH, BASE_VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_HEIGHT
            if self.isFullScreen():
                #self.resizeVideo(QSize(BASE_VIDEO_WIDTH, BASE_VIDEO_HEIGHT))
                self.showNormal()
                self.resize(BASE_VIDEO_WIDTH, BASE_VIDEO_HEIGHT)
                # self.resize(self.sizeHint())
            else:
                self.showFullScreen()
                #self.resizeVideo(self.webCamLabel.size())


class WebCamWorker(QObject):
    finished = pyqtSignal()
    changePixmap = pyqtSignal(QImage)

    def init(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.mode = 'MULTI'
        self.button = None
        self.selected_box_cls = None

    def run(self):
        ''' Main function that process stream from webcam '''
        ret = False
        while not ret:
            ret, frame = self.cap.read()

        frame = cv2.flip(frame, 1)

        tracks, cls_names = tracker.tracking(frame)

        if self.button == 'LEFT':
            self.selected_box_cls, bbox = self.check_boxes(tracks, cls_names, frame)
            # single_tracker.init(frame, bbox)

        if self.mode == 'SINGLE':
            success, tracks = single_tracker.update(frame)
            if not success:
                self.mode = 'MULTI'

        frame = tracker.draw_boxes(frame, tracks, cls_names, self.mode, self.selected_box_cls)

        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(VIDEO_WIDTH, VIDEO_HEIGHT, Qt.KeepAspectRatio)
        self.changePixmap.emit(p.copy())



    @pyqtSlot(str, QPoint)
    def handle_mouse_click(self, button, click_point=None):
        ''' Receives signal from mouse about which button have been pressed and at which point'''
        if button == 'LEFT' and self.mode == 'MULTI' or button == 'RIGHT' and self.mode == 'SINGLE':
            self.button = button
            self.click_point = click_point
            if self.button == 'RIGHT':
                self.mode = 'MULTI'
                self.button = None


    def check_boxes(self, tracks, cls_names, frame):
        ''' Checks if mouse button has been press inside the object bbox
        Parameters:
            tracks - bbox received from DeepSort
            cls_names - names corresponding to tracks
            frame - frame from webcam stream
        Returns:
            closest_bbox_name - name of object that has been selected
            bbox - coordinate of selected bbox '''
        x, y = self.click_point.x(), self.click_point.y()
        closest_bbox_name = -1
        min_dist = float('inf')
        bbox = None
        for track, cls in zip(tracks, cls_names):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            xc = x1 + (x2-x1)//2
            yc = y1 + (y2-y1)//2

            if x1 <= x <= x2 and y1 <= y <= y2 and (x - xc)**2 + (y - yc)**2 < min_dist:
                closest_bbox_name = cls
                min_dist = (x - xc)**2 + (y - yc)**2
                x1, y1, w, h = np.asarray(track.to_tlwh(), dtype=int)
                bbox = [max(x1, 0), max(y1, 0), min(w, VIDEO_WIDTH - x1 - 1), min(h, VIDEO_HEIGHT - y1 - 1)]


        self.button = None

        if closest_bbox_name != -1:
            self.mode = 'SINGLE'
            single_tracker.init(frame, bbox)

        return closest_bbox_name, bbox

    @pyqtSlot()
    def finish_app(self):
        ''' Closes webcam '''
        self.cap.release()


app = QApplication([])
win = Window()
win.show()
app.exec_()
