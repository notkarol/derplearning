#!/usr/bin/env python3

import tensorflow as tf
import torch

class Model:
    
    def __init__(self, log, path):
        """
        Open the model
        """
        self.log = log
        self.path = path

    def __del__(self):
        """
        Deconstructor to close file objects
        """
        pass

    
    def evaluate(self, frame, speed, steer):
        """ 
        Cut out the patch and run the model on it
        """
        return speed, steer
