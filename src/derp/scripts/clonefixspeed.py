#!/usr/bin/env python3

import cv2
import numpy as np
from os.path import join
import torch
from torch.autograd import Variable
import derp.util as util
from derp.inferer import Inferer

class CloneFixSpeed(Inferer):

    def __init__(self, hw_config, sw_config, model_dir=None, nocuda=False):

        self.hw_config = hw_config
        self.sw_config = sw_config
        self.model_dir = model_dir
        self.nocuda = nocuda
        self.exp = 'clone'
        
        # Prepare the input camera
        self.component_name = self.sw_config[self.exp]['patch']['component']
        for component in self.hw_config['components']:
            if component['name'] == self.component_name:
                self.hw_component = component

        # Prepare camera inputs
        self.bbox = util.get_patch_bbox(self.hw_component, sw_config[self.exp])
        self.size = (sw_config[self.exp]['patch']['width'], sw_config[self.exp]['patch']['height'])

        # Prepare model
        if self.model_dir is not None:
            self.model_path = join(model_dir, 'clone.pt')
            self.model = torch.load(self.model_path)
            self.model.eval()
        else:
            self.model_path = None
            self.model = None


    def prepare_x(self, state):
        frame = state[self.component_name]
        patch = frame[self.bbox.y : self.bbox.y + self.bbox.h,
                      self.bbox.x : self.bbox.x + self.bbox.w]
        thumb = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)
        return thumb

    
    def prepare_batch(self, thumb):
        batch = np.reshape(thumb, [1] + list(thumb.shape)).transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).float()
        if not self.nocuda:
            batch = batch.cuda()
        batch /= 255
        return batch

    
    def plan(self, state):
        if self.model is None:
            return 0.0, 0.0
        
        thumb = self.prepare_x(state)
        batch = self.prepare_batch(thumb)
        
        out = self.model(Variable(batch))

        if self.nocuda:
            predictions = out.data.numpy()[0]
        else:
            predictions = out.data.cpu().numpy()[0]

        # Figure out speed and steer. Speed is fixed based on state
        speed = state['speed_offset']
        steer = (self.sw_config[self.exp]['params']['curr'] * float(predictions[0]) +
                 self.sw_config[self.exp]['params']['prev'] * state['steer'])
        return speed, steer
