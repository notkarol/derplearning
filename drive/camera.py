#!/usr/bin/env python3

import cv2
from datetime import datetime
import numpy as np
import os
import re
import sys
from time import strftime, gmtime, time

'''
Defines the camera class
Function list:
  init
  del
  getFrame
  record
  snapshot
  discoverCamera
'''

class Camera:
    
    def __init__(self, log, index=None, width=640, height=480, fps=20):
        """
        Open a camera capture
        """
        self.log = log
        self.index = self.discoverCamera() if index is None else index
        self.width = width
        self.height = height
        self.fps = fps

        # Log Arguments
        log.config('camera_%i_fps' % self.index, self.fps)
        log.config('camera_%i_height' % self.index, self.height)
        log.config('camera_%i_width' % self.index, self.width)
            
        # Input  video
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Output video in log
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.extension = 'mp4'
        self.writers = {}

        # Initialize known cameras
        views = ['front']
        for view in views:
            path = os.path.join(log.folder, "%s.%s" % (view, self.extension))
            self.writers[view] = cv2.VideoWriter(path, self.fourcc, self.fps,
                                                 (self.width, self.height))

        
    def __del__(self):
        """
        Make sure we leave the camera for another program
        """
        self.cap.release()

        
    def getFrame(self):
        """
        Read the next camera frame. This is a blocking call
        """
        # Get time and steering angle
        ret, frame = self.cap.read()

        # If we can't get a frame don't return any and set a warning
        if not ret:
            sys.stderr.write("Failed to get frame!")
            return None
        
        return frame

    
    def record(self, camera, frame):
        self.writers[camera].write(frame)


    def snapshot(self, view, timestamp, frame):
        name = os.path.join(self.log.folder, "%s-%.6f.png" % (view, timestamp))
        cv2.imwrite(name, frame)
        
        
    def discoverCamera(self,
                       last_available=True):  # use most recently plugged in camera
        """
        Find available cameras
        """
        devices = [int(f[-1]) for f in sorted(os.listdir('/dev')) if re.match(r'^video[0-9]', f)]
        if len(devices) == 0:
            return None
        device = devices[-1] if last_available else devices[0]
        return device
