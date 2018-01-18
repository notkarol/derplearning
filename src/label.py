import numpy as np
import cv2
import os
import sys
import time
import yaml
import argparse
import derp.util
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from skimage.draw import line_aa
import derp.util as util

'''
TODO: Make the viewer and the graph different windows to simplify
display choices.
'''

class Labeler(object):

    def legal_position(self, pos):
        return 0 <= pos < self.n_frames
    

    def update_label(self, id1, id2, marker):
        if not marker:
            return

        # Update the labels that are to be stored
        beg, end = min(id1, id2), max(id1, id2)
        #The update is applied inclusive to both ends of the id range
        for i in range(beg, end + 1):
            self.labels[i] = marker

        # Update the visual label bar
        beg_pos, end_pos = self.frame_pos(beg), self.frame_pos(end)
        self.label_bar[beg_pos: end_pos + 1] = self.marker_color[marker]
        
    
    def seek(self, frame_id):
        if not self.legal_position(frame_id):
            print("seek failed illegal target:", frame_id)
            return False

        self.update_label(frame_id, self.frame_id, self.marker)

        self.frame_id = frame_id - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
        print("%i %5i %6.3f %6.3f" % (self.frame_id, self.timestamps[self.frame_id],
                                      self.speeds[frame_id], self.steers[frame_id]))
        self.read()
        self.show = True
        return True

    
    def read(self):
        if not self.legal_position(self.frame_id + 1):
            print("read failed illegal", self.frame_id)
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
        self.window[self.fh:, :, :] = self.black

        
    def draw_bar_status(self):
        self.window[self.fh : self.fh + int(self.bhh // 10), self.fwi, :] = self.label_bar
        

    def draw_bar_zeroline(self):
        self.window[self.fh + self.bhh, self.fwi, :] = self.gray75

    
    def draw_graph(self, data_vector, color):
        #interpolate the data vector to fill in gaps
        d_interpld = interp1d(self.state_x, data_vector)
        
        #convert data vector to a data array the size of the window's x dimension
        data_bar = np.array([-d_interpld(x) * self.bhh + 0.5 for x in self.window_x],
                                  dtype=np.int)

        # All locations where we need to draw lines
        data_jump_locs = []

        for loc in np.where(abs(data_bar[:-1] - data_bar[1:]) >= 2)[0]:
            rr, cc, val= line_aa(data_bar[loc] + self.fh + self.bhh, loc,
                                 data_bar[loc + 1] + self.fh + self.bhh, loc + 1)
            data_jump_locs.append( (rr, cc) )

        """ Draw the speed and steering lines on the bar below the video. 
        Takes about 1ms to run through the for loops"""
        self.window[data_bar + self.fh + self.bhh, self.fwi, :] = color
        for rr, cc in data_jump_locs:
            self.window[rr, cc, :] = color
        
    
    #This function updates the viewer calling all appropriate functions
    def display(self):
        
        # Update the pixels of the frame
        self.window[:self.frame.shape[0], :, :] = self.frame

        self.draw_bar_blank()
        self.draw_bar_timemarker()
        self.draw_bar_zeroline()
        self.draw_bar_status()
        self.draw_graph(data_vector=self.speeds, color=self.cyan)
        self.draw_graph(data_vector=self.steers, color=self.green)

        if self.model:
            self.draw_graph(data_vector=self.m_speeds, color=self.blue)
            self.draw_graph(data_vector=self.m_steers, color=self.red)

        # Display the newly generated window
        cv2.imshow('Labeler %s' % self.recording_path, self.window)


    def save_labels(self):
        with open(self.labels_path, 'w') as f:
            f.write("timestamp,status\n")
            for timestamp, label in zip(self.timestamps, self.labels):
                f.write("%.6f,%s\n" % (timestamp, label))
        print("Saved labels at ", self.labels_path)

    #Handles keyboard inputs during data labeling process.
    def handle_input(self):
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27: # escape
            return False
        elif key == ord('p') or key == ord(' '):
            self.paused = not self.paused
        elif key == ord('g'):
            self.marker = 'good'
            self.show = True
        elif key == ord('r'):
            self.marker = 'risk'
            self.show = True
        elif key == ord('t'):
            self.marker = 'junk'
            self.show = True
        elif key == ord('c'):
            self.marker = ''
            self.show = True
        elif key == ord('s'):
            self.save_labels()
        elif key == 82: # up
            self.seek(self.frame_id + self.fps)
        elif key == 84: # down
            self.seek(self.frame_id - self.fps)
        elif key == 81: # left
            self.seek(self.frame_id - 1)
        elif key == 83: # right
            self.seek(self.frame_id + 1)
        elif key == ord('`'):
            self.seek(0)
        elif ord('1') <= key <= ord('9'):
            self.seek(int(self.n_frames * (key - ord('0')) / 10))
        elif key == ord('0'):
            self.seek(self.n_frames - 1)

        elif key != 255:
            print("Unknown key press: [%s]" % key)
            
        return True
    

    def frame_pos(self, frame_id):
        return min(self.fw - 1, int(frame_id / len(self.timestamps) * self.fw))        


    def init_labels(self):
        self.labels_path = os.path.join(self.recording_path, 'label.csv')
        if os.path.exists(self.labels_path):
            _, _, self.labels = derp.util.read_csv(self.labels_path, floats=False)
            for i in range(len(self.labels)):
                self.labels[i] = self.labels[i][0]                           
        else:
            self.labels = ["" for _ in range(self.n_frames)]
        
                
    def init_states(self):
        self.state_path = os.path.join(self.recording_path, 'state.csv')
        self.timestamps, self.state_headers, self.states = derp.util.read_csv(self.state_path)
        self.speeds = self.states[:, self.state_headers.index('speed')]
        self.steers = self.states[:, self.state_headers.index('steer')]

            
    def init_camera(self):
        self.video_path = os.path.join(self.recording_path, 'camera_front.mp4')
        self.cap = cv2.VideoCapture(self.video_path)
        self.n_frames = min(len(self.timestamps), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_id = -1
        self.read()

        
    def init_window(self):
        self.fh = self.frame.shape[0]
        self.fw = self.frame.shape[1]
        self.bhh = 150 # bar half height

        self.fwi = np.arange(self.fw)   #frame width index
        self.window_shape = list(self.frame.shape)
        self.window_shape[0] += self.bhh* 2 + 1
        self.window = np.zeros(self.window_shape, dtype=np.uint8)

        #state_x is a vector containing the indicies of the x coordinate of the state graph
        self.state_x = np.linspace(0, 1, len(self.timestamps) )
        self.window_x = np.linspace(0, 1, self.fw)

        self.paused = True
        self.show = False
        self.marker = ''
        self.label_bar = np.ones((self.fw, 3), dtype=np.uint8) * self.gray50
        for i in range(self.n_frames):
            self.update_label(i, i, self.labels[i])    
        

    #Create arrays of predicted speed and steering using the designated model
    def predict(self, config, model_path):
        #Initialize the model output data vectors:
        self.m_speeds = np.zeros(self.n_frames, dtype=float)
        self.m_steers = np.zeros(self.n_frames, dtype=float)

        #opens the video config file
        video_config = util.load_config('%s/%s' % (self.recording_path, 'config.yaml') )
        bot = Inferer(  video_config = video_config, 
                        model_config = config,
                        folder = self.recording_path,
                        model_path = model_path)

        #Move the capture function to the start of the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, -1)

        for i in range(self.n_frames ):
            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("read failed frame", frame_id)

            if i%1==0:
                (self.m_speeds[i], 
                 self.m_steers[i],
                 batch) = bot.evaluate(frame, 
                                    self.timestamps[i], 
                                    config, 
                                    model_path)

            else:
                self.m_speeds[i] = self.m_speeds[i-1]
                self.m_steers[i] = self.m_steers[i-1]

        #Restore the camera position to wherever it was before predict was called
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)


    def run_labeler(self, config_path=None, model_path=None):
        
        #create driving predictions:
        self.model = None
        if model_path:
            self.model = model_path
            config = util.load_config(path=config_path)
            self.predict(config_path, model_path)

        #Start the display window
        self.display()
        
        # Loop through video
        while self.cap.isOpened():
            if not self.paused and self.frame_id < self.n_frames:
                self.update_label(self.frame_id, self.frame_id, self.marker)
                self.read()
                self.show = True

            if self.show:
                self.display()
                self.show = False
                
            if not self.handle_input():
                break

        
    def __init__(self, recording_path, scale=1):
        
        self.scale = scale #Image Scale Factor
        self.recording_path = recording_path

        # Variables useful for later
        self.cap = None
        # Useful color variables:
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
        self.marker_color = { '' : self.gray50,
                              'good': self.green,
                              'risk': self.orange,
                              'junk': self.red}

        #Initialize all elements needed to label data.
        self.init_states()
        self.init_camera()
        self.init_labels()
        # Initialze the viewer window
        self.init_window()


    def __del__(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="recording path location")
    parser.add_argument('--scale', type=float, default=1.0, help="frame rescale ratio")
    parser.add_argument('--config', type=str, default=None, help="physical configuration")
    parser.add_argument('--infer', type=str, default=None, help="infer configuration")

    args = parser.parse_args()
    
    labeler = Labeler(recording_path=args.path, scale=args.scale)
    labeler.run_labeler(args.config, args.infer)
