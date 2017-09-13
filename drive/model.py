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

    #define model output characteristics:
    self.n_lines = 3
    self.n_points = 3
    self.n_dimensions = 2

    #define camera characteristics
    #linear measurements given in mm
    self.camera_height = 380
    self.camera_min_view = 500 #Fixme remeasure distance
    #arcs measured in radians
    self.camera_to_ground_arc = np.arctan(camera_min_view/camera_height)
    self.camera_vert_arc = 60 * (pi/180)
    self.camera_horz_arc = 80 * (pi/180)

      
      
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
    return batch
              
  def evaluate(self, frame, speed, steer):
    """ 
    Cut out the patch and run the model on it
    """
    batch = self.preprocess(frame)
    nn_speed, nn_steer = self.model.predict_on_batch(batch)[0]
    return nn_speed, nn_steer, batch[0]

  #Funtion which calls the learning network and outputs road line control points
  def road_spotter(self, frame):

    batch = self.preprocess(frame)

    road_lines = self.model.predict_on_batch(batch)[0]

    return np.reshape(road_lines, (self.n_lines, self.n_dimensions, self.n_points) )

  #This function uses a transform to map the percieved road onto a 2d plane beneath the car
  def road_mapper(self, frame):

    road_spots = road_spotter(frame)

    road_map = np.zeros(np.shape(road_spots), np.float)

    #First we deal with all the points
    z = self.camera_height * np.tan(self.camera_to_ground_arc + road_map[:, 0, :] * self.camera_vert_arc/ self.source_size[1])


