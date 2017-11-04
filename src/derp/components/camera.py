#!/usr/bin/env python3

import cv2
import numpy as np
import os
import re
import sys
from time import time
from derp.component import Component

class Camera(Component):

    def __init__(self, config, name, index=None):
        super(Camera).__init__()
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.cap = None
        self.video = None
        self.timestamp_fp = None

        
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.timestamp_fp is not None:
            self.timestamp_fp.close()
            self.timestamp_fp = None


    def act(self):
        return True
    

    def discover(self):
        """
        Find available cameras, use most recently plugged in camera
        """
        # Find camera index
        if self.index is None:
            devices = [int(f[-1]) for f in sorted(os.listdir('/dev'))
                       if re.match(r'^video[0-9]', f)]
            if len(devices) == 0:
                self.connected = False
                return self.connected
            self.index = devices[-1]
        
        # Connect to camera
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FPS, self.config[name]['frame']['fps'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config[name]['frame']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config[name]['frame']['height'])
        self.connected = True
        return self.connected
        
        
    def folder(self, folder):

        # Prepre video writer
        if self.video is not None:
            self.video.release()
        self.video_path = os.path.join(folder, "%s.mp4" % self.name)
        self.video = cv2.VideoWriter(self.video_path, self.fourcc, self.config['frame']['fps'],
                                     (self.config['frame']['width'], self.config['frame']['height']))

        # Prepare timestamp writer
        if self.timestamp_fp is not None:
            self.timestamp_fp.close()
        self.out_csv_path = os.path.join(folder, "%s.csv" % self.name)
        self.out_csv_fp = open(self.timestamp_path, 'w')

        return True
    
        
    def sense(self, state):

        # Make sure we have a camera open
        if self.cap is None:
            return False
        
        # Read the next video frame
        ret = self.cap.grab()
        timestamp = int(time() * 1E6)
        frame = self.cap.retrieve()

        # If we can't get a frame then return a blank one
        if not ret:
           frame = np.zeros((self.config['frame']['height'], self.config['frame']['width'],
                              self.config['frame']['depth']), np.uint8)

        # Updat the state and return whet
        state[self.name] = (timestamp, frame)

        # Update buffer
        self.out_buffer.append((timestamp, frame))
        
        return ret


    def write(self):

        if self.video is None or self.out_csv_fp is None:
            return False

        for row in self.out_buffer:
            timestamp, frame = row
            self.out_csv_fp.write(str(timestamp) + "\n")
            self.video.write(frame)
        self.out_csv_fp.flush()
        self.out_buffer = []
            
        return True

