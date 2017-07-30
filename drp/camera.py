#!/usr/bin/env python3

import cv2
import numpy as np

class Camera:

    def __init__(self, index=0):
        self.index = index

        # Open video
        self.cap = cv2.VideoCapture(self.index)

    def __del__(self):
        self.cap.release()
        
    def getFrame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
