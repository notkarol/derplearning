#!/usr/bin/env python3

import io
import numpy as np
import os
import re
import select
import sys
from time import time, sleep
import v4l2capture
import PIL.Image
from derp.component import Component
import derp.util as util
import subprocess

class Camera(Component):

    def __init__(self, config):
        super(Camera, self).__init__(config)
        self.cap = None
        self.frame_counter = 0
        self.start_time = 0
        self.config = config


    def __del__(self):
        if self.cap is not None:
            self.cap.close()
            self.cap = None


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
        try:
            self.cap = v4l2capture.Video_device("/dev/video%i" % self.index)
        except FileNotFoundError:
            print("Camera index [%i] not found" % self.index)
            self.cap = None
            
        self.connected = self.cap is not None
        if not self.connected:
            return self.connected

        # start the camerea
        w, h = self.cap.set_format(self.config['width'], self.config['height'], fourcc='MJPG') # YUYV
        fps = self.cap.set_fps(self.config['fps'])
        self.cap.create_buffers(1)
        self.cap.queue_all_buffers()
        self.cap.start()

        # Return whether we have succeeded
        return True


    def scribe(self, state):

        # If we're not recording, make sure we also don't have to encode mp4s
        if not state['record']:
            if self.folder:
                fps = int(self.frame_counter / (time() - self.start_time) + 0.5)
                cmd = " ".join(['gst-launch-1.0',
                                  'multifilesrc',
                                  'location="%s/%s/%%06d.jpg"' % (self.folder, self.config['name']),
                                  '!', '"image/jpeg,framerate=%i/1"' % fps, 
                                  '!', 'jpegparse',
                                  '!', 'jpegdec',
                                  '!', 'omxh264enc', 'bitrate=8000000',
                                  '!', '"video/x-h264, stream-format=(string)byte-stream"',
                                  '!', 'h264parse',
                                  '!', 'mp4mux',
                                  '!', 'filesink location="%s/%s.mp4"' % (self.folder,
                                                                          self.config['name'])])
                subprocess.Popen(cmd, shell=True)
                self.folder = None
            return True
                
        # Create directory for storing images
        if state['folder'] != self.folder:
            self.folder = state['folder']
            self.recording_dir = os.path.join(self.folder, self.config['name'])
            os.mkdir(self.recording_dir)
            self.frame_counter = 0
            self.start_time = time()


        # Write the frame
        self.write()
        return True


    def sense(self, state):
        
        # Make sure we have a camera open
        if self.cap is None:
            return False
        
        # Read the next video frame. If we couldn't get it, use the last one
        frame = None
        counter = 1
        while frame is None and counter:
            counter -= 1
            select.select((self.cap,), (), ())
            try:
                image_data = self.cap.read_and_queue()
                frame = np.array(PIL.Image.open(io.BytesIO(image_data)))
            except:
                print("Camera: Unable to get frame. Retrying")
            
            
        # Update the state and our out buffer
        timestamp = int(time() * 1E6)
        state['timestamp'] = timestamp
        state[self.config['name']] = frame

        if state['record']:
            self.out_buffer.append((timestamp, image_data))
        return True


    def write(self):

        for timestamp, image_data in self.out_buffer:
            path = '%s/%06i.jpg' % (self.recording_dir, self.frame_counter)
            with open(path, 'wb') as f:
                f.write(image_data)
            self.frame_counter += 1
            
        del self.out_buffer[:]
            
        return True

