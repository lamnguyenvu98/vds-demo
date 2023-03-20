import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import GLib, Gst, GstVideo

import numpy as np
import torch
import time
import cv2
from model import detector

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

class VideoWidget(QThread):
    send_result_frame = Signal(np.ndarray, int)
    send_draw_frame = Signal(np.ndarray)
    
    def __init__(self, parent, filename, index):
        super().__init__()
        self.pipeline = None
        self.filename = filename
        self.index = index
        self.pr = parent
        self.draw_frame = None

    def run(self):
        global detector
        Gst.init(None)

        # create a gstreamer pipeline with videotestsrc element
        self.pipeline = Gst.parse_launch(f"""
            filesrc location={self.filename} ! 
            decodebin ! 
            videoconvert ! 
            video/x-raw,format=RGBA ! 
            appsink name=sink""")

        # get the app sink element
        self.appsink = self.pipeline.get_by_name("sink")

        # set some properties on the app sink
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("max-buffers", 1)

        # connect to the new-sample signal of the app sink
        self.appsink.connect("new-sample", self.on_new_sample)

        bus = self.pipeline.get_bus()
        # bus.set_sync_handler(self.__on_message)
        # bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect("message", self.__on_message)
        bus.connect("sync-message::element", self.__on_sync_message)

        # start playing the pipeline
        self.play()

    def __gst_to_np(self, sample):
        '''
        Converts gst to numpy ndarray
        '''
        buf = sample.get_buffer()
        caps = sample.get_caps()
        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
             caps.get_structure(0).get_value('width'),
             4),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr[..., :3]

    def get_draw_frame(self):
        return self.draw_frame

    def on_new_sample(self, appsink):
        # self.spin(0.2)
        # get the sample from the app sink
        sample = appsink.emit("pull-sample")

        img = self.__gst_to_np(sample)
        self.send_draw_frame.emit(img)
        with torch.no_grad():
            print('Thread : {} - memory identity: {}'.format(self.index, id(detector)))
            results = detector(img.copy())
        
        frames = self.postprocessing(results, img.astype(np.uint8))
        
        self.send_result_frame.emit(frames, self.index)
        return Gst.FlowReturn.OK

    def postprocessing(self, results, frame):
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cord = results.xyxyn[0][:, :-1].cpu().numpy()
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # If score is less than 0.3 we avoid making a prediction.
            if row[4] < 0.3:
                continue
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
            bgr = (0, 255, 0)  # color of the box
            classes = detector.names  # Get the name of label index
            label_font = cv2.FONT_HERSHEY_COMPLEX  # Font for the label.
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
            cv2.putText(frame, classes[int(labels[i])], (x1, y1), label_font, 1, bgr, 2)
        return frame

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QApplication.processEvents()

    def __on_message(self, bus, message):
        '''
        '''

        t = message.type
        if t == Gst.MessageType.EOS:
            self.stop()
            black = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.send_frame(black, self.index)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err} ", debug)
            self.stop()
            black = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.send_frame(black, self.index)
        
        return Gst.BusSyncReply.PASS

    def __on_sync_message(self, bus, message):
        '''
        '''

        if message.get_structure().get_name() == 'prepare-window-handle':
            win_id = self.windowId
            imagesink = message.src
            imagesink.set_property("force-aspect-ratio", True)
            # if not window id then create new window
            if win_id is None:
                win_id = self.movie_window.get_property('window').get_xid()
            imagesink.set_window_handle(win_id)

    def pause(self):
        self.pipeline.set_state(Gst.State.PAUSED)
    
    def play(self):
        self.pipeline.set_state(Gst.State.PLAYING)
    
    def stop(self):
        self.pipeline.set_state(Gst.State.STOP)

    def taskStop(self):
        self.stop()
        self.quit()
        self.pr.image_labels.pop(self.index)
        self.pr.threads.pop(self.index)
        self.pr.is_maximized.pop(self.index)