#!/usr/bin/env python3

import tensorflow as tf
import torch
from keras.models import model_from_json

class Model:
    
    def __init__(self, log, model_path, weights_path):
        """
        Open the model
        """
        self.log = log
        self.path = path

        with open(model_path) as f:
            json_contents = f.read()
            self.model = model_from_json(json_contents)
        self.model.load_weights(weights_path)
        
    def __del__(self):
        """
        Deconstructor to close file objects
        """
        pass

    
    def evaluate(self, frame, speed, steer):
        """ 
        Cut out the patch and run the model on it
        """
        out = self.model.predict(frame, batch_size=1, verbose=1)
        return speed, steer, out
