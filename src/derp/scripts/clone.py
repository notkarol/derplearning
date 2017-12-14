#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from derp.component import Component
import derp.util
import derp.imagemanip

class Clone(Component):

    def __init__(self, config, full_config):
        super(Clone, self).__init__(config, full_config)
        
        self.config = config
        self.no_cuda = 'no_cuda' in full_config and full_config['no_cuda']
        
        # Which config is our settings coming from
        self.source_config = derp.util.find_component_config(full_config, config['camera_name'])

        # Prepare camera inputs
        self.bbox = derp.imagemanip.get_patch_bbox(self.config['thumb'],
                                                   self.source_config)
        self.size = (config['thumb']['width'], config['thumb']['height'])

        # Prepare model
        self.model = None
        if 'model_path' in config and config['model_path'] is not None:
            model_path = derp.util.find_matching_file(config['model_path'], '\.pt$')
            if model_path is not None:
                self.model = torch.load(model_path)
                self.model.eval()

        # Useful variables for params
        self.prev_steer = 0
        self.prev_speed = 0


    # Prepare input image
    def prepare_thumb(self, state):
        import cv2
        frame = state[self.config['camera_name']]
        patch = frame[self.bbox.y : self.bbox.y + self.bbox.h,
                      self.bbox.x : self.bbox.x + self.bbox.w]
        thumb = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)
        return thumb


    # Prepare status
    def prepare_status(self, state):
        status = np.zeros(len(self.config['status']), dtype=np.float32)
        for i, sd in enumerate(self.config['status']):
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
        if not self.no_cuda:
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
        predictions = out.data.numpy()[0] if self.no_cuda else out.data.cpu().numpy()[0]

        # Normalize predictions to desired range
        for i, pd in enumerate(self.target_config['predict']):
            predictions[i] /= pd['scale']

        return predictions


    def plan(self, state):
        # Do not do anything if we do not have a loaded model
        if self.model is None:
            return 0.0, 0.0

        # Get the predictions of our model
        predictions = self.predict(state)

        # Use the given speed and steer directly from predictions
        speed = float(predictions[0])
        steer = float(predictions[1])

        return speed, steer
