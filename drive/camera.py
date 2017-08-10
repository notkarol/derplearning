#!/usr/bin/env python3

import cv2
from datetime import datetime
import numpy as np
import os
import re
import sys
from time import strftime, gmtime, time

class Camera:
    
    def __init__(self, index=None):
        self.width = 640
        self.height = 480
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')

        # Get camera
        if index is not None:
            self.index = index
        else:
            self.index = self.discoverCamera()

        # Input  video
        self.cap = cv2.VideoCapture(self.index)
        #self.cap.set(cv2.CAP_PROP_FPS, 30)
        #self.cap.set(cv2.CAP_PROP_FOURCC, self.fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Output video
        self.out_vid = None
        self.out_csv = None
        self.folder = None
        
        
    def __del__(self):
        self.stopRecording()
        self.cap.release()

        
    def getFrame(self):

        # Get time and steering angle
        timestamp = "%.3f" % time()
        ret, frame = self.cap.read()

        # If we can't get a frame don't return any and set a warning
        if not ret:
            sys.stderr.write("Failed to get frame at %s\n" % timestamp)
            return None

        # Otherwise record it
        if self.out_vid:
            cv2.imwrite(os.path.join(self.folder, "%s.png" % timestamp), frame)
            self.out_vid.write(frame)
            self.out_csv.write(",".join([timestamp]) + "\n")
            self.out_csv.flush()
        
        return frame

    def startRecording(self):
        """
        Create a folder and output video file
        """
        
        # Make sure we're not already recording. If we are, start a new file
        if self.out_vid:
            self.stopRecording()

        # Create folder
        self.folder = strftime('%Y%m%dT%H%M%SZ', gmtime())
        os.mkdir(self.folder)

        # Create video
        out_path = os.path.join(self.folder, 'video%i.avi' % (self.index))
        videotimes_path = os.path.join(self.folder, 'video.csv')
        
        self.out_vid = cv2.VideoWriter(out_path, self.fourcc, self.fps, (self.width, self.height), False)
        self.out_csv = open(videotimes_path, 'w')


    def stopRecording(self):
        """
        Release the video file and clear variables
        """
        if self.out_vid is not None:
            self.out_vid.release()
            self.out_vid = None
            self.out_csv.close()
            self.out_csv = None
        self.folder = None


    def discoverCamera(self,
                       last_available=True):  # use most recently plugged in camera
        """
        Find available cameras
        """
        devices = [int(f[-1]) for f in os.listdir('/dev') if re.match(r'^video[0-9]', f)]
        if len(devices) == 0:
            return None
        device = devices[-1] if last_available else devices[0]
        print("Using camera [%i]" % device)
        return device
