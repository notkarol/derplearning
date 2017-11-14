#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import derp.util as util
from derp.inferer import Inferer

class Clone(Inferer):

    def __init__(self, hw_config, sw_config, model_dir, state):

        self.hw_config = hw_config
        self.sw_config = sw_config
        self.model_dir = model_dir

        # Prepare the input camera
        self.component_name = self.sw_config['clone']['patch']['component']
        for component in self.hw_config['components']:
            if component['name'] == self.component_name:
                self.hw_component = component

        # Prepare camera inputs
        self.bbox = util.get_patch_bbox(self.hw_component, sw_config['clone'])
        self.size = (sw_config['clone']['patch']['width'], sw_config['clone']['patch']['height'])

        # Prepare model
        self.model_path = os.path.join(model_dir, 'clone.pt')
        self.model = torch.load(self.model_path)
        self.model.eval()

        
    def plan(self, state):

        # Prepare input thumbnail
        frame = state[self.component_name]
        patch = frame[self.bbox.y : self.bbox.y + self.bbox.h,
                      self.bbox.x : self.bbox.x + self.bbox.w]
        thumb = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)

        # Prepare batch
        batch = np.reshape(thumb, [1] + list(thumb.shape)).transpose((0, 3, 1, 2))
        batch = torch.from_numpy(batch).float()
        batch = batch.cuda()
        batch /= 255

        # get predictions
        out = self.model(Variable(batch))
        predictions = out.data.cpu().numpy()[0]

        # Store the data we're getting if we're debugging
        if state['record'] and self.sw_config['debug']:
            cv2.imwrite(os.path.join(state['folder'], "%i_frame.png" % state['timestamp']), frame)
            cv2.imwrite(os.path.join(state['folder'], "%i_patch.png" % state['timestamp']), patch)
            cv2.imwrite(os.path.join(state['folder'], "%i_thumb.png" % state['timestamp']), thumb)


        # Desired upodates
        proposal = {field: float("%.6f" % val) for field, val in
                    zip(self.sw_config['clone']['predict'], predictions)}
        return proposal
