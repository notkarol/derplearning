#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import derpmodels
import derputil

class Model:
    
    def __init__(self, config, folder, model_path):
        """
        Open the model
        """
        self.config = config
        self.folder = folder
        self.model_path = model_path

        self.bbox = derputil.getPatchBbox(self.config, self.config, perspective='drive')
        self.size = derputil.getPatchSize(self.config)
        self.model = torch.load(self.model_path)
        
        

    def evaluate(self, frame, timestamp, speed, steer):
        """ 
        Cut out the patch and run the model on it
        """
        patch = frame[self.bbox.y : self.bbox.y + self.bbox.h,
                      self.bbox.x : self.bbox.x + self.bbox.w]
        thumb = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)
        batch = np.reshape(thumb, [1] + list(thumb.shape)).transpose((0, 3, 1, 2))
        batch_cuda = torch.from_numpy(batch).float().cuda() / 255
        batch_var = Variable(batch_cuda)
        predictions = self.model(batch_var).data.cpu().numpy()[0]
        
        if self.config['debug']:
            cv2.imwrite(os.path.join(self.folder, "%i_frame.png" % timestamp), frame)
            cv2.imwrite(os.path.join(self.folder, "%i_patch.png" % timestamp), patch)
            cv2.imwrite(os.path.join(self.folder, "%i_thumb.png" % timestamp), thumb)
        

        return (predictions[self.config['states'].index('speed')],
                predictions[self.config['states'].index('steer')], batch[0])
