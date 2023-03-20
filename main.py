import sys
from PySide6.QtCore import Slot, QEvent, Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QWidget, QGridLayout, QLabel, QSizePolicy, 
                               QMainWindow, QMessageBox, QApplication)
import cv2
import numpy as np
from video_thread import VideoWidget

# /home/pep/Drive/PCLOUD/Projects/youtube-scraping/data/001.mp4

class MultiScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.display_width = 640
        self.display_height = 480

        self.layout = QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.url_list = np.loadtxt('list_cam.txt', dtype=np.object_).flatten().tolist()

        self.image_labels = list()
        self.threads = list()
        self.is_maximized = {}
        self.num_cols = 2
        
        for i in range(len(self.url_list)):
            label = QLabel()
            object_name = 'cam_{}'.format(i)
            label.setObjectName(object_name)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            label.setScaledContents(True)
            label.installEventFilter(self)
            self.is_maximized[object_name] = {'label_index': i, 'state': False}
            self.image_labels.append(label)
            self.layout.addWidget(label, i//self.num_cols, i%self.num_cols)
            thread = VideoWidget(self, self.url_list[i], i)
            thread.send_result_frame.connect(self.update_image)
            self.threads.append(thread)
            thread.start()
        
        self.setLayout(self.layout)
    
    @Slot(np.ndarray, int)
    def update_image(self, cv_img, index):
        """Updates the image_label with a new numpy array"""
        # print(cv_img)
        qt_img = self.convert_cv_qt(cv_img)
        self.image_labels[index].setPixmap(qt_img)
        
    def convert_cv_qt(self, cv_img):
        """Convert from an numpy array to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def eventFilter(self, source, event) -> bool:
        def filter_obj(ignore_name, show=False):
            for k, v in self.is_maximized.items():
                if k == ignore_name:
                    continue
                index = v['label_index']
                if show:
                    self.image_labels[index].show()
                else:
                    self.image_labels[index].hide()
        
        if event.type() == QEvent.MouseButtonDblClick:
            obj_name_clicked = source.objectName()
            if not self.is_maximized[obj_name_clicked]['state']:
                filter_obj(ignore_name=obj_name_clicked, show=False)
                self.is_maximized[obj_name_clicked]['state'] = True
            else:
                filter_obj(ignore_name=obj_name_clicked, show=True)
                self.is_maximized[obj_name_clicked]['state'] = False

            return True
        else:
            return super().eventFilter(source, event)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.multi_screen = MultiScreen()
        self.setMinimumSize(640, 480)
        self.setCentralWidget(self.multi_screen)

    def closeEvent(self, event) -> None:
        ret = QMessageBox.information(self,
            "Quit CCTV", # title
            "Are you sure to Quit?", # content
            QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            for thread in self.multi_screen.threads:
                if thread.isRunning():
                    thread.taskStop()
                    del thread
            event.accept()
        else:
            event.ignore()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())