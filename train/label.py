import numpy as np
import cv2
import os
import sys
import time
import derputil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Labeler(object):

    def seek(self, frame_id):
        if frame_id < 0 or frame_id >= len(self.states):
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        return True

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False
        self.frame = frame
        self.frame_id += 1
        return True

    def display(self):
        
        # Update the pixels of the frame
        self.window[:self.frame.shape[0], :, :] = self.frame

        # Zero out and redraw position line
        self.window[self.frame.shape[0]:, self.frame_pos, 2] = 0
        self.frame_pos = int(self.frame_id / len(self.timestamps) * self.frame.shape[1])
        self.window[self.frame.shape[0]:, self.frame_pos, 2] = 255

        # Draw zero line
        self.window[self.frame.shape[0] + self.bhh + 1, :, :] = 128

        # Draw speet and steer
        for i in range(self.frame.shape[1]):
            speed = self.speed_bar[i] + self.frame.shape[0] + self.bhh + 1
            steer = self.steer_bar[i] + self.frame.shape[0] + self.bhh + 1
            self.window[speed, i, 0] = 255
            self.window[steer, i, 1] = 255

        # Display window
        cv2.imshow('Labeler %s' % self.recording_path, self.window)

        
    def handle_input(self):
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            return False

        return True
        
        
    def __init__(self, recording_path):

        self.recording_path = recording_path
        self.video_path = os.path.join(self.recording_path, 'camera_front.mp4')
        self.state_path = os.path.join(self.recording_path, 'state.csv')

        # Open camera
        self.cap = cv2.VideoCapture(self.video_path)

        # Open States
        self.timestamps, self.state_dicts = derputil.readState(self.state_path)
        self.speeds = np.zeros(len(self.timestamps), dtype=np.float)
        self.steers = np.zeros(len(self.timestamps), dtype=np.float)
        for pos, d in enumerate(self.state_dicts):
            self.speeds[pos] = d['speed']
            self.steers[pos] = d['steer']
        
        # Prepare extra room
        self.window_shape = None
        self.window = None
        self.frame_id = 0
        self.frame_pos = 0

        # Loop through video
        while self.cap.isOpened():

            # read in a frame
            if not self.read():
                break

            # If we haven't yet initialized a window, do so
            if self.window_shape is None:

                # Prepare window
                self.bhh = 100 # bar half height
                self.window_shape = list(self.frame.shape)
                self.window_shape[0] += self.bhh* 2 + 1
                self.window = np.zeros(self.window_shape, dtype=np.uint8)

                old_x = np.linspace(0, 1, len(self.speeds))
                new_x = np.linspace(0, 1, self.window_shape[1])
                speed_f = interp1d(old_x, self.speeds)
                steer_f = interp1d(old_x, self.steers) 
                self.speed_bar = np.array([speed_f(x) * self.bhh + 0.5 for x in new_x], dtype=np.int)
                self.steer_bar = np.array([steer_f(x) * self.bhh + 0.5for x in new_x], dtype=np.int)


                # Draw speet and steer
                for i in range(self.frame.shape[1]):
                    speed = self.speed_bar[i] + self.frame.shape[0] + self.bhh + 1
                    steer = self.steer_bar[i] + self.frame.shape[0] + self.bhh + 1
                    self.window[speed, i, 0] = 255
                    self.window[steer, i, 1] = 255

                
            # display frame and menubars
            self.display()

            # Check if there's any input from the user
            if not self.handle_input():
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    recording_path = sys.argv[1]
    Labeler(recording_path)
