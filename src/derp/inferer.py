#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import derp.models
import derp.util as util

def clone_train():
    return transforms.Compose([transforms.ColorJitter(brightness=0.5,
                                                      contrast=0.5,
                                                      saturation=0.5,
                                                      hue=0.1),
                               transforms.ToTensor()])

def clone_eval():
    def reflective(x):
        return x[:2]
    return reflective

class Inferer:
    
    def __init__(self, video_config, model_config, folder, model_path):
        """
        Open the model
        """
        self.model_config = model_config
        self.video_config = video_config
        self.folder = folder
        self.model_path = model_path

        self.bbox = util.getPatchBbox(self.video_config, self.model_config)
        self.size = util.getPatchSize(self.model_config)
        self.model = torch.load(self.model_path)
        self.model.eval()
        self.model.mode = 'clone'

    def evaluate(self, frame, timestamp, speed, steer):
        """ 
        Cut out the patch and run the model on it
        """
        patch = frame[self.bbox.y : self.bbox.y + self.bbox.h,
                      self.bbox.x : self.bbox.x + self.bbox.w ]
        thumb = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)
        batch = np.reshape(thumb, [1] + list(thumb.shape)).transpose((0, 3, 1, 2))
        batch_cuda = torch.from_numpy(batch).float().cuda() / 255
        batch_var = Variable(batch_cuda)
        predictions = self.model(batch_var).data.cpu().numpy()[0]

        # Store the data we're getting
        if self.model_config['debug']:
            cv2.imwrite(os.path.join(self.folder, "%i_frame.png" % timestamp), frame)
            cv2.imwrite(os.path.join(self.folder, "%i_patch.png" % timestamp), patch)
            cv2.imwrite(os.path.join(self.folder, "%i_thumb.png" % timestamp), thumb)
        
        if self.model.mode == 'lines':
            road_spots = np.reshape(predictions, (1,
                                                  self.model_config['n_lines'],
                                                  self.model_config['n_dimensions'],
                                                  self.model_config['n_points']))
            nn_speed = speed
            nn_steer = 3 * road_spots[0, 1, 0, 0] 
            #center_vector = road_spots[1, :, 1] - road_spots[1, :, 0]
            #nn_steer = nn_steer + 2 * nn_speed * center_vector[0] / center_vector[1]
        elif self.model.mode == 'clone':
            nn_speed = predictions[self.model_config['states'].index('speed')]
            nn_steer = predictions[self.model_config['states'].index('steer')]
        return (nn_speed, nn_steer, batch[0])
                
