#!/usr/bin/env python3

import cv2
import numpy as np
import os
import torch
from derp.component import Component
import derp.util

class Clone(Component):

    def __init__(self, config, full_config, state):
        super(Clone, self).__init__(config, full_config, state)
        self.camera_config = derp.util.find_component_config(full_config, config['camera_name'])

        # Show the user what we're working with
        derp.util.print_image_config(self.camera_config)
        derp.util.print_image_config(self.config['thumb'])
        
        # Prepare camera inputs
        self.bbox = derp.util.get_patch_bbox(self.config['thumb'], self.camera_config)
        self.size = (config['thumb']['width'], config['thumb']['height'])

        # Prepare model
        self.model = None
        if 'model_dir' in full_config and full_config['model_dir'] is not None:
            model_path = derp.util.find_matching_file(full_config['model_dir'], 'clone.pt$')
            if model_path is not None:
                self.model = torch.load(model_path)
                self.model.eval()

        # Useful variables for params
        self.prev_steer = 0
        self.prev_speed = 0

        # Data saving
        self.out_buffer = []
        self.frame_counter = 0  


    def prepare_thumb(self, state):
        frame = state[self.config['camera_name']]
        patch = derp.util.crop(frame, self.bbox)
        thumb = derp.util.resize(patch, self.size)
        return thumb


    def predict(self, state):
        status = derp.util.extractList(self.config['status'], state)
        thumb = self.prepare_thumb(state)
        status_batch = derp.util.prepareVectorBatch(status)
        thumb_batch = derp.util.prepareImageBatch(thumb)
        status_batch = derp.util.prepareVectorBatch(status)
        if self.model:
            prediction_batch = self.model(thumb_batch, status_batch)
            prediction = derp.util.unbatch(prediction_batch)
            derp.util.unscale(self.config['predict'], prediction)
        else:
            prediction = np.zeros(len(self.config['predict']), dtype=np.float32)
            # Debugging
            #cv2.imshow('frame', frame)
            #cv2.imshow('patch', patch)
            #cv2.imshow('thumb', thumb)
            #cv2.waitKey(1)
            
        # Store the thumb and our prediction
        if self.is_recording(state):
            self.out_buffer.append((state['timestamp'], thumb, prediction))
        return prediction


    def plan(self, state):
        prediction = self.predict(state)
        return prediction

    
    def record(self, state):

        # If we can not record, return false
        if not self.is_recording(state):
            return False

        # If we are initialized, then spit out jpg images directly to disk
        if not self.is_recording_initialized(state):
            super(Clone, self).record(state)
            self.folder = state['folder']
            self.recording_dir = os.path.join(self.folder, self.config['name'])
            self.frame_counter = 0
            os.mkdir(self.recording_dir)

        # Write out buffered images
        for timestamp, thumb, prediction in self.out_buffer:
            path = '%s/%06i.jpg' % (self.recording_dir, self.frame_counter)
            cv2.imsave(path, thumb)
            self.frame_counter += 1
            # TODO handle predictions
        del self.out_buffer[:]

        return True                         
