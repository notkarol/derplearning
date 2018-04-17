#!/usr/bin/env python3
import cv2
import numpy as np
import os
import re
import select
import sys
from time import time, sleep
import v4l2capture
import subprocess
from derp.component import Component
import derp.util as util

class Camera(Component):

    def __init__(self, config, full_config, state):
        super(Camera, self).__init__(config, full_config, state)

        self.cap = None
        self.out_buffer = []
        self.frame_counter = 0
        self.start_time = 0
        
        # Find available cameras, use most recently plugged in camera
        if self.config['index'] is None:
            devices = [int(f[-1]) for f in sorted(os.listdir('/dev'))
                       if re.match(r'^video[0-9]', f)]
            if len(devices) == 0:
                self.connected = False
                return self.connected
            self.index = devices[-1]
        else:
            self.index = self.config['index']

        # Connect to camera, exit if we can't
        try:
            self.cap = v4l2capture.Video_device("/dev/video%i" % self.index)
        except FileNotFoundError:
            print("Camera index [%i] not found" % self.index)
            return

        # start the camerea
        w, h = self.cap.set_format(self.config['width'], self.config['height'], fourcc='MJPG') # YUYV
        fps = self.cap.set_fps(self.config['fps'])
        self.cap.create_buffers(1)
        self.cap.queue_all_buffers()
        self.cap.start()
        self.state[self.config['name']] = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Return whether we have succeeded
        self.ready = True


    def __del__(self):
        super(Camera, self).__del__()
        if self.cap is not None:
            self.cap.close()
            self.cap = None


    def sense(self):
        
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
                self.state['timestamp'] = time()
                image_bytes = self.cap.read_and_queue()
                image_array = np.fromstring(image_bytes, np.uint8)
                self.state[self.config['name']][:,:,:] = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print("Camera: Unable to get frame. Retrying")
                print(e)
        
        # Append frame to out buffer if we're writing
        if self.is_recording():
            self.out_buffer.append((self.state['timestamp'], image_bytes))
        return True


    def record(self):
        # Do not write if write is not desired.. but close up any recording process by
        # trying to encode an mp4 in the background.
        if not self.is_recording():
            if self.is_recording_initialized():
                fps = int(self.frame_counter / (time() - self.start_time) + 0.5)
                args = (self.folder, self.config['name'])
                cmd = " ".join(['gst-launch-1.0',
                                'multifilesrc',
                                'location="%s/%s/%%06d.jpg"' % args,
                                '!', '"image/jpeg,framerate=%i/1"' % fps, 
                                '!', 'jpegparse',
                                '!', 'jpegdec',
                                '!', 'omxh264enc', 'bitrate=8000000',
                                '!', '"video/x-h264, stream-format=(string)byte-stream"',
                                '!', 'h264parse',
                                '!', 'mp4mux',
                                '!', 'filesink location="%s/%s.mp4"' % args])
                subprocess.Popen(cmd, shell=True)
                self.folder = None
            return True                

        # If we are initialized, then spit out jpg images directly to disk
        if not self.is_recording_initialized():
            super(Camera, self).record()
            self.folder = self.state['folder']
            self.recording_dir = os.path.join(self.folder, self.config['name'])
            self.frame_counter = 0
            self.start_time = time()
            os.mkdir(self.recording_dir)

        # Write out buffered images
        for timestamp, image_data in self.out_buffer:
            path = '%s/%06i.jpg' % (self.recording_dir, self.frame_counter)
            with open(path, 'wb') as f:
                f.write(image_data)
            self.frame_counter += 1
        del self.out_buffer[:]

        return True
