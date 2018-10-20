"""
The Camera component manages the camera interface.
"""
import cv2
import numpy as np
import os
import re
import select
import sys
from time import time, sleep
import subprocess
from derp.component import Component
import derp.util as util

class Camera(Component):
    """
    The Camera component manages the camera interface.
    """

    def __init__(self, config, state):
        super(Camera, self).__init__(config, state)

        self.cap = None
        self.frame_counter = 0
        self.start_time = 0
        self.image_bytes = b''
        self.state[self.config['name']] = None
        self.__connect()
        self.width = int(self.config['width'] * self.config['resize'] + 0.5)
        self.height = int(self.config['height'] * self.config['resize'] + 0.5)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def __connect(self):
        if self.cap:
            del self.cap
            self.cap = None
        self.ready = self.__find()
        if self.ready:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        return self.ready
            
    def __find(self):
        """ Finds and connects to the camera """
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
            return False
        return True
            
    def sense(self):
        """ Read the next video frame. If we couldn't get it, use the previous one """
        if not self.ready:
           self.__connect()

        if self.ready:
            frame = None
            ret, frame = self.cap.read()
            if ret:
                frame = util.resize(frame, (self.width, self.height))
                sensor_name = self.config['name']
                self.state[sensor_name] = frame
                if self.state['debug']:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)                
            else:
                print("Camera: Unable to get frame")
                self.ready = False
        return self.ready

