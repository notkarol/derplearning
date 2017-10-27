#!/usr/bin/env python3

import cv2
import numpy as np
import os
import re
import sys
from time import strftime, gmtime, time

class Camera:
    
    def __init__(self, config, index=None):
        """
        Open a camera capture
        """
        self.index = self.discoverCamera() if index is None else index
        self.config = config

        # Input video
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FPS, self.config['frame']['fps'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame']['height'])

        # Output video in log
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.rec = None

        
    def __del__(self):
        """
        Make sure we leave the camera for another program
        """
        self.cap.release()
        if self.rec is not None:
            self.rec.release()


    def record(self, folder):
        """
        Initialize recording
        """
        path = os.path.join(folder, "camera_front.mp4")
        self.rec = cv2.VideoWriter(path, self.fourcc, self.config['frame']['fps'],
                                   (self.config['frame']['width'], self.config['frame']['height']))
        
        
    def read(self):
        """
        Read the next camera frame. 
        """

        ret, frame = self.cap.read()

        # If we can't get a frame then return a blank one
        if not ret or frame is None:
            shape = (self.config['frame']['height'],
                     self.config['frame']['width'],
                     self.config['frame']['depth'])
            frame = np.zeros(shape, np.uint8)
            print("CAMERA.READ: Failed to get frame from self.cap")
            
        return frame

    
    def write(self, frame):
        if self.rec is not None:
            self.rec.write(frame)
        else:
            print("CAMERA.WRITE: Failed due to uninitialized self.rec")

        
    def discoverCamera(self, last_available=True):  
        """
        Find available cameras, use most recently plugged in camera
        """
        devices = [int(f[-1]) for f in sorted(os.listdir('/dev')) if re.match(r'^video[0-9]', f)]
        if len(devices) == 0:
            return None
        device = devices[-1] if last_available else devices[0]
        return device
