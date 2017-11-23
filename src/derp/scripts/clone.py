#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import derp.util 
import derp.imagemanip

class Clone():

    def __init__(self, source_hw_config, target_hw_config, sw_config, path, nocuda):

        self.sw_config = sw_config
        self.target_hw_config = target_hw_config
        self.source_hw_config = source_hw_config
        self.nocuda = nocuda

        # If there is not a different source hw config, use the target one
        if self.source_hw_config is None:
            self.source_hw_config = self.target_hw_config

        # Which component our patch comes from
        self.component_name = self.sw_config['thumb']['component']
        for component in self.target_hw_config['components']:
            if component['name'] == self.component_name:
                self.target_hw_component = component
        for component in self.source_hw_config['components']:
            if component['name'] == self.component_name:
                self.source_hw_component = component

        # Prepare camera inputs
        self.bbox = derp.imagemanip.get_patch_bbox(self.target_hw_component,
                                                   self.source_hw_component)
        self.size = (sw_config['thumb']['width'],
                     sw_config['thumb']['height'])

        # Prepare model
        self.model = None
        if path is not None:
            model_path = derp.util.find_matching_file(path, '\.pt$')
            if model_path is not None:
                self.model = torch.load(model_path)
                self.model.eval()

        # Useful variables for params
        self.prev_steer = 0
        self.prev_speed = 0


    # Prepare input image
    def prepare_thumb(self, state):
        frame = state[self.component_name]
        patch = frame[self.bbox.y : self.bbox.y + self.bbox.h,
                      self.bbox.x : self.bbox.x + self.bbox.w]
        thumb = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)
        return thumb


    # Prepare status
    def prepare_status(self, state):
        status = np.zeros(len(self.sw_config['status']), dtype=np.float32)
        for i, sd in enumerate(self.sw_config['status']):
            status[i] = state[sd['field']] * sd['scale']
        return status
    

    # Prepare input batch
    def prepare_batch(self, thumbs, statuses):

        # If we're given a single image make it a 4d batch
        if len(thumbs.shape) == 3:
            new_shape = [1] + list(thumbs.shape)
            thumbs = np.reshape(thumbs, new_shape)

        if len(statuses.shape) == 1:
            new_shape = [1] + list(statuses.shape)
            statuses = np.reshape(statuses, new_shape)
            
        # Change from HWD to DHW
        thumbs = thumbs.transpose((0, 3, 1, 2))

        # Convert to torch batch
        thumbs_batch = torch.from_numpy(thumbs).float()
        statuses_batch = torch.from_numpy(statuses).float()

        # Convert to cuda if we need to
        if not self.nocuda:
            thumbs_batch = thumbs_batch.cuda()
            statuses_batch = statuses_batch.cuda()

        # Normalize batch
        thumbs_batch /= 255

        # Return as variable
        return Variable(thumbs_batch), Variable(statuses_batch)


    def predict(self, state):
        thumb = self.prepare_thumb(state)
        status = self.prepare_status(state)
        thumb_batch, status_batch = self.prepare_batch(thumb, status)
        out = self.model(thumb_batch, status_batch)

        # Prepare predictions from model output by converting them to numpy vector
        if self.nocuda:
            predictions = out.data.numpy()[0]
        else:
            predictions = out.data.cpu().numpy()[0]

        # Normalize predictions to desired range
        for i, pd in enumerate(self.sw_config['predict']):
            predictions[i] /= pd['scale']

        return predictions


    def plan(self, state):
        # Do not do anything if we do not have a loaded model
        if self.model is None:
            return 0.0, 0.0

        # Get the predictions of our model
        predictions = self.predict(state)

        # Prepare parameters variale for verbosity
        params = self.sw_config['params']

        # Speed is what the model says, but possibly averaged
        speed = float(predictions[0])
        speed = params['speed_curr'] * speed + params['speed_prev'] * self.prev_speed

        # Steer is what the model says, but possibly averaged
        steer = float(predictions[1])
        steer = params['steer_curr'] * steer + params['steer_prev'] * self.prev_steer

        return speed, steer
