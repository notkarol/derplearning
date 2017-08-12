#!/usr/bin/env python3

import cv2
import tensorflow as tf
import numpy as np
from keras.models import model_from_json

class Model:
    
    def __init__(self, log, model_path, weights_path):
        """
        Open the model
        """
        self.log = log
        self.model_path = model_path
        self.weights_path = weights_path

        self.source_size = (640, 480)
        self.crop_size = (640, 160)
        self.crop_x = 0
        self.crop_y = self.source_size[1] - self.crop_size[1]
        self.target_size = (128, 32)
        
        with open(model_path) as f:
            json_contents = f.read()
            self.model = model_from_json(json_contents)
        self.model.load_weights(weights_path)
        
        
    def __del__(self):
        """
        Deconstructor to close file objects
        """
        pass

    def preprocess(self, example):
        patch = example[self.crop_y : self.crop_y + self.crop_size[1],
                        self.crop_x : self.crop_x + self.crop_size[0], :]
        thumb = cv2.resize(patch, self.target_size)
        batch = np.reshape(thumb, [1] + list(thumb.shape))
        return np.array(batch, dtype=np.float32)
                
    def evaluate(self, frame, speed, steer):
        """ 
        Cut out the patch and run the model on it
        """
        example = self.preprocess(frame)
        nn_speed, nn_steer = self.model.predict_on_batch(example)[0]
        return nn_speed, nn_steer

                    
