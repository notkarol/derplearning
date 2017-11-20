#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import derp.util as util

class CameraClone():

    def __init__(self, sw_config, target_hw_config, source_hw_config, path, nocuda):

        self.sw_config = sw_config
        self.target_hw_config = target_hw_config
        self.source_hw_config = source_hw_config
        self.nocuda = nocuda

        # If there is not a different source hw config, use the target one
        if self.source_hw_config is None:
            self.source_hw_config = self.target_hw_config

        # Which component our patch comes from
        self.component_name = self.sw_config['patch']['component']
        for component in self.target_hw_config['components']:
            if component['name'] == self.component_name:
                self.target_hw_component = component
        for component in self.source_hw_config['components']:
            if component['name'] == self.component_name:
                self.source_hw_component = component

        # Prepare camera inputs
        self.bbox = util.get_patch_bbox(self.target_hw_component,
                                        self.source_hw_component,
                                        self.sw_component['patch'])
        self.size = (sw_config['patch']['width'],
                     sw_config['patch']['height'])

        # Prepare model
        self.model = None
        if self.path is not None:
            model_path = util.find_matching_file(self.path, 'pt$')
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


    # Prepare input batch
    def prepare_batch(self, thumbs):

        # If we're given a single image make it a 4d batch
        if len(thumbs.shape) == 3:
            new_shape = [1] + list(thumb.shape)
            thumbs = np.reshape(thumb, new_shape)

        # Change from HWD to DHW
        thumbs = thumbs.transpose((0, 3, 1, 2))

        # Convert to torch batch
        batch = torch.from_numpy(thumbs).float()

        # Convert to cuda if we need to
        if not self.nocuda:
            batch = batch.cuda()

        # Normalize batch
        batch /= 255

        # Return as variable
        return Variable(batch)


    def predict(self, state):
        thumb = self.prepare_thumb(state)
        batch = self.prepare_batch(thumb)
        out = self.model(batch)

        if self.nocuda:
            predictions = out.data.numpy()[0]
        else:
            predictions = out.data.cpu().numpy()[0]
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
