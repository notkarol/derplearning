#!/usr/bin/env python3

import cv2
import numpy as np
import os
import re
import sys
from time import time
from derp.component import Component

class Camera(Component):

    def __init__(self, config, name):
        super(Camera, self).__init__(config, name)
        self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.cap = None
        self.video = None

        
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video is not None:
            self.video.release()
            self.video = None
        if self.out_csv_fp is not None:
            self.out_csv_fp.close()
            self.out_csv_fp = None


    def act(self, state):
        return True
    

    def discover(self):
        """
        Find available cameras, use most recently plugged in camera
        """
        # Find camera index
        if self.config['index'] is None:
            devices = [int(f[-1]) for f in sorted(os.listdir('/dev'))
                       if re.match(r'^video[0-9]', f)]
            if len(devices) == 0:
                self.connected = False
                return self.connected
            self.index = devices[-1]
        else:
            self.index = self.config['index']

        # Connect to camera
        self.cap = cv2.VideoCapture(self.index)        
        self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])

        # Return whether we have succeeded
        self.connected = self.cap.open(self.index)
        return self.connected
        
        
    def scribe(self, state):
        if not state['folder'] or state['folder'] == self.folder:
            return False
        self.folder = state['folder']
        
        # Prepre video writer
        if self.video is not None:
            self.video.release()
        self.video_path = os.path.join(self.folder, "%s.mp4" % self.name)
        self.video = cv2.VideoWriter(self.video_path, self.fourcc, self.config['fps'],
                                     (self.config['width'], self.config['height']))

        # Prepare timestamp writer
        if self.out_csv_fp is not None:
            self.out_csv_fp.close()
        self.out_csv_path = os.path.join(self.folder, "%s.csv" % self.name)
        self.out_csv_fp = open(self.out_csv_path, 'w')
        return True
    
        
    def sense(self, state):
        
        # Make sure we have a camera open
        if self.cap is None:
            return False
        
        # Read the next video frame
        ret, frame = self.cap.read()

        # Update the state and our out buffer
        if ret:
            timestamp = int(time() * 1E6)
            state['timestamp'] = timestamp
            state[self.name] = frame

            if state['record']:
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
        del self.out_buffer[:]
            
        return True

