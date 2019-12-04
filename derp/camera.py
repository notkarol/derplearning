"""
The Camera component manages the camera interface.
"""
import capnp
import camera_capnp
import cv2
import os
import re

import derp.util

class Camera:
    """
    The Camera component manages the camera interface.
    """

    def __init__(self, config):
        self.config = config
        self.cap = None
        self.frame_counter = 0
        self.start_time = 0
        self.image_bytes = b''
        self.__connect()
        if 'resize' not in self.config:
            source_config['resize'] = 1
        self.width = int(self.config['width'] * self.config['resize'] + 0.5)
        self.height = int(self.config['height'] * self.config['resize'] + 0.5)
        self.publisher = derp.util.publisher(self.config['name'])
        
    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def __connect(self):
        if self.cap:
            del self.cap
            self.cap = None
        if self.config['index'] is None:
            devices = [int(f[-1]) for f in sorted(os.listdir('/dev'))
                       if re.match(r'^video[0-9]', f)]
            if len(devices) == 0:
                self.connected = False
                return self.connected
            self.index = devices[-1]
        else:
            self.index = self.config['index']

        # Connect to camera, exit if we can't
        try:
            self.cap = cv2.VideoCapture(self.index)
        except:
            print("Camera index [%i] not found. Failing." % self.index)
            self.cap = None
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def message(self, frame):
        jpg_buffer = cv2.imencode('.jpg', frame)[1].tostring()
        msg = camera_capnp.Camera.new_message(
            timestamp=derp.util.get_timestamp(),
            yaw=self.config['yaw'],
            pitch=self.config['pitch'],
            roll=self.config['roll'],
            x=self.config['x'],
            y=self.config['y'],
            z=self.config['z'],
            height=self.config['height'],
            width=self.config['width'],
            depth=self.config['depth'],
            hfov=self.config['hfov'],
            vfov=self.config['vfov'],
            fps=self.config['fps'],
            jpg=jpg_buffer)
        return msg
        
    def read(self):
        """ Read the next video frame. If we couldn't get it, use the previous one """
        ret, frame = self.cap.read()
        if ret:
            frame = derp.util.resize(frame, (self.width, self.height))
            return frame
        return None

    def run(self):
        while True:
            frame = self.read()
            if frame is None:
                self.__connect()
                return None
            frame = derp.util.resize(frame, (self.width, self.height))
            msg = self.message(frame)
            self.publisher.send_multipart([self.config['name'], msg.to_bytes()])
