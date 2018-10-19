#!/usr/bin/env python3
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

    def __init__(self, config, state):
        super(Camera, self).__init__(config, state)

        self.cap = None
        self.frame_counter = 0
        self.start_time = 0
        self.image_bytes = b''
        self.state[self.config['name']] = None
        self.__connect()

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
            height, width = self.config['height'], self.config['width']
            resize, recrop = self.config['resize'], self.config['recrop']
            if ret:
                if isinstance(recrop, list) and len(recrop) == 4 and recrop != [0, 0, 1, 1]:
                    x = int(width * recrop[0])
                    y = int(height * recrop[1])
                    width = int(width * recrop[2])
                    height = int(height * recrop[3])
                    frame = util.crop(frame, bbox=util.Bbox(x, y, width, height))
                if 0 < resize < 1:
                    width = int(width * resize)
                    height = int(height * resize)
                    frame = util.resize(frame, (width, height))
                if self.state['debug']:
                    cv2.imshow('resize', frame)
                    cv2.waitKey(2)
                sensor_name = self.config['name']
                self.state[sensor_name] = frame
            else:
                print("Camera: Unable to get frame")
                self.ready = False
        return self.ready

