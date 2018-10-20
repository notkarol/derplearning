#!/usr/bin/env python3

import cv2
import numpy as np
import os
import torch
from derp.controller import Controller
import derp.util

class Clone(Controller):

    def __init__(self, config, car_config, state):
        super(Clone, self).__init__(config, car_config, state)
        self.camera_config = derp.util.find_component_config(car_config,
                                                             config['thumb']['component'])

        # Show the user what we're working with
        derp.util.print_image_config('Source', self.camera_config)
        derp.util.print_image_config('Target', self.config['thumb'])
        
        # Prepare camera inputs
        self.bbox = derp.util.get_patch_bbox(self.config['thumb'], self.camera_config)
        self.size = (config['thumb']['width'], config['thumb']['height'])

        # Prepare model
        self.model_dir = derp.util.get_controller_models_path(self.config['name'])
        self.model_path = derp.util.find_matching_file(self.model_dir, 'clone.pt$')
        if self.model_path is not None and os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
            self.model.eval()
        else:
            self.model = None
            print("Clone: Unable to find model path [%s]" % self.model_path)

        # Useful variables for params
        self.prev_steer = 0
        self.prev_speed = 0

        # Data saving
        self.frame_counter = 0  
 
    def prepare_thumb(self):
        frame = self.state[self.config['thumb']['component']]
        if frame is not None:
            patch = derp.util.crop(frame, self.bbox)
            thumb = derp.util.resize(patch, self.size)
            if 'debug' in self.state and self.state['debug']:
                cv2.imshow('patch', patch)
                cv2.waitKey(1)
        else:
            dim = [self.config['thumb']['height'],
                   self.config['thumb']['width']]
            if self.config['thumb']['depth'] > 1:
                dim += [self.config['thumb']['depth']]
            thumb = np.zeros(dim, dtype=np.float32)
        return thumb

    def predict(self):
        status = derp.util.extractList(self.config['status'], self.state)
        self.state['thumb'] = self.prepare_thumb()
        status_batch = derp.util.prepareVectorBatch(status)
        thumb_batch = derp.util.prepareImageBatch(self.state['thumb'])
        status_batch = derp.util.prepareVectorBatch(status)
        if self.model:
            prediction_batch = self.model(thumb_batch, status_batch)
            prediction = derp.util.unbatch(prediction_batch)
            derp.util.unscale(self.config['predict'], prediction)
        else:
            prediction = np.zeros(len(self.config['predict']), dtype=np.float32)
        self.state['prediction'] = prediction
        

    def plan(self):
        self.predict()
        if self.state['auto']:
            self.state['speed'] = float(self.state['prediction'][0])
            self.state['steer'] = float(self.state['prediction'][1])
