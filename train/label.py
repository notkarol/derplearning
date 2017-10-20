import numpy as np
import cv2
import os
import sys
import time
import derputil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Labeler(object):

    def legal_position(self, pos):
        return 0 <= pos < self.n_frames
    

    def update_label(self, id1, id2, marker):

        # Update the labels that are to be stored
        if not marker:
            beg, end = min(id1, id2), max(id1, id2)
            self.labels[beg : end + 1] = marker

        # Update the visual label bar
        beg_pos, end_pos = self.frame_pos(beg), self.frame_pos(end)
        self.label_bar[beg_pos: end_pos + 1] = self.marker_color[marker]
        
    
    def seek(self, frame_id):
        if not self.legal_position(frame_id):
            print("seek failed illegal", frame_id)
            return False

        self.update_label(frame_id, self.frame_id, self.marker)

        self.frame_id = frame_id - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        self.show = True
        return True

    
    def read(self):
        if not self.legal_position(self.frame_id + 1):
            print("read failed illegal", frame_id)
            return False

        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            print("read failed frame", frame_id)
            return False

        # Resize frame as needed
        self.frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                                interpolation=cv2.INTER_AREA)
        self.frame_id += 1
        return True


    
    def draw_bar_timemarker(self):
        self.window[self.fh:, self.frame_pos(self.frame_id), :] = self.marker_color[self.marker]

        
    def draw_bar_blank(self):
        self.window[self.frame.shape[0]:, :, :] = self.black


    def draw_bar_zeroline(self):
        midpoint = self.frame.shape[0] + self.bhh + 1
        self.window[midpoint, :, :] = self.gray50

        
    def draw_bar_speed_steer(self):
        self.window[self.speed_bar + self.fh + self.bhh + 1, self.fwi, :] = self.cyan
        self.window[self.steer_bar + self.fh + self.bhh + 1, self.fwi, :] = self.magenta
        
        
    def display(self):
        
        # Update the pixels of the frame
        self.window[:self.frame.shape[0], :, :] = self.frame

        self.draw_bar_blank()
        self.draw_bar_timemarker()
        self.draw_bar_zeroline()
        self.draw_bar_speed_steer()
        
        # Display window
        cv2.imshow('Labeler %s' % self.recording_path, self.window)

        
    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF

        if key == 27: # escape
            return False
        elif key == ord('p'):
            self.paused = not self.paused
        elif key == ord('q'):
            self.marker = 'good'
            self.show = True
        elif key == ord('w'):
            self.marker = 'risk'
            self.show = True
        elif key == ord('e'):
            self.marker = 'junk'
            self.show = True
        elif key == ord('r'):
            self.marker = ''
            self.show = True
        elif key == 82:
            self.seek(self.frame_id + self.fps)
        elif key == 84:
            self.seek(self.frame_id - self.fps)
        elif key == 81:
            self.seek(self.frame_id - 1)
        elif key == 83:
            self.seek(self.frame_id + 1)
        elif key != 255:
            print("Unknown key press: [%s]" % key)
            
        return True
    

    def frame_pos(self, frame_id):
        return min(self.fw - 1, int(frame_id / len(self.timestamps) * self.fw))        

    
    def init_states(self):
        self.state_path = os.path.join(self.recording_path, 'state.csv')
        self.timestamps, self.state_dicts = derputil.readState(self.state_path)
        self.speeds = np.zeros(len(self.timestamps), dtype=np.float)
        self.steers = np.zeros(len(self.timestamps), dtype=np.float)
        for pos, d in enumerate(self.state_dicts):
            self.speeds[pos] = d['speed']
            self.steers[pos] = d['steer']

            
    def init_camera(self):
        self.video_path = os.path.join(self.recording_path, 'camera_front.mp4')
        self.cap = cv2.VideoCapture(self.video_path)
        self.n_frames = min(len(self.timestamps), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_id = 0
        self.read()

        
    def init_window(self):
        self.bhh = 100 # bar half height
        self.fh = self.frame.shape[0]
        self.fw = self.frame.shape[1]
        self.fwi = np.arange(self.fw)
        self.window_shape = list(self.frame.shape)
        self.window_shape[0] += self.bhh* 2 + 1
        self.window = np.zeros(self.window_shape, dtype=np.uint8)

        self.state_x = np.linspace(0, 1, len(self.timestamps))
        self.window_x = np.linspace(0, 1, self.fw)
        speed_f = interp1d(self.state_x, self.speeds)
        steer_f = interp1d(self.state_x, self.steers) 
        self.speed_bar = np.array([speed_f(x) * self.bhh + 0.5 for x in self.window_x],
                                  dtype=np.int)
        self.steer_bar = np.array([steer_f(x) * self.bhh + 0.5 for x in self.window_x],
                                  dtype=np.int)
        
        self.paused = True
        self.show = False
        self.marker = ''
        self.labels = ['' for _ in range(self.n_frames)]
        self.label_bar = np.zeros((self.fw, 3), dtype=np.uint8)

        
    def __init__(self, recording_path):
        self.recording_path = recording_path


        # Variables useful for later
        self.scale = 0.5
        self.cap = None
        self.red = np.array([0, 0, 255], dtype=np.uint8)
        self.green = np.array([32, 192, 32], dtype=np.uint8)
        self.blue = np.array([255, 128, 0], dtype=np.uint8)
        self.yellow = np.array([0, 255, 255], dtype=np.uint8)
        self.cyan = np.array([255, 255, 0], dtype=np.uint8)
        self.magenta = np.array([255, 0, 255], dtype=np.uint8)
        self.gray25 = np.array([64, 64, 64], dtype=np.uint8)
        self.gray50 = np.array([128, 128, 128], dtype=np.uint8)
        self.gray75 = np.array([192, 192, 192], dtype=np.uint8)
        self.black = np.array([0, 0, 0], dtype=np.uint8)
        self.white = np.array([255, 255, 255], dtype=np.uint8)
        self.orange = np.array([255, 128, 0], dtype=np.uint8)
        self.purple = np.array([255, 0, 128], dtype=np.uint8)
        self.marker_color = { '' : self.white,
                              'good': self.green,
                              'risk': self.orange,
                              'junk': self.red}
        
        # Initialze 
        self.init_states()
        self.init_camera()
        self.init_window()
        self.display()
        
        # Draw speet and steer
        for i in range(self.frame.shape[1]):
            speed = self.speed_bar[i] + self.frame.shape[0] + self.bhh + 1
            steer = self.steer_bar[i] + self.frame.shape[0] + self.bhh + 1
            self.window[speed, i, 0] = 255
            self.window[steer, i, 1] = 255
        
        # Loop through video
        while self.cap.isOpened():
            if not self.paused or self.show:
                if not self.read():
                    break
                self.display()
                self.show = False
            if not self.handle_input():
                break


    def __del__(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    recording_path = sys.argv[1]
    Labeler(recording_path)
