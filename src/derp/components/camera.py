#!/usr/bin/env python3

import io
import numpy as np
import os
import re
import select
import sys
from time import time
import v4l2capture
import PIL.Image
from derp.component import Component
import derp.util as util

class Camera(Component):

    def __init__(self, config):
        super(Camera, self).__init__(config)
        self.cap = None


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
        self.cap.create_buffers(30)
        self.cap.queue_all_buffers()
        self.cap.start()

        # Return whether we have succeeded
        return True


    def scribe(self, state):
        if not state['folder'] or state['folder'] == self.folder:
            self.write()
            return False

        # Create directory for storing images
        self.folder = state['folder']
        self.recording_dir = os.path.join(self.folder, self.config['name'])
        os.mkdir(self.recording_dir)

        self.write()
        return True


    def sense(self, state):
        
        # Make sure we have a camera open
        if self.cap is None:
            return False
        
        # Read the next video frame
        select.select((self.cap,), (), ())
        image_data = self.cap.read_and_queue()
        frame = np.array(PIL.Image.open(io.BytesIO(image_data)))

        # Update the state and our out buffer
        timestamp = int(time() * 1E6)
        state['timestamp'] = timestamp
        state[self.config['name']] = frame

        if state['record']:
            self.out_buffer.append((timestamp, image_data))
        
        return True


    def write(self):

        for timestamp, image_data in self.out_buffer:
            path = '%s/%i.jpg' % (self.recording_dir, timestamp)
            with open(path, 'wb') as f:
                f.write(image_data)
            
        del self.out_buffer[:]
            
        return True

