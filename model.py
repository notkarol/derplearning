#!/usr/bin/env python3

import cv2
import tensorflow as tf
import numpy as np
from keras.models import model_from_json, model_from_yaml

'''
Defines the model class containing the following functions:
  __init__
  __del__
  evaluate
  roadspotter
  road_mapper
  pilot_mk1
'''

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

    # Temporary
    self.source_size = (1920, 1080)
    self.crop_size = (1440, 360)
    self.crop_x = 240
    self.crop_y = 720
    self.target_size = (128, 64)

    if model_path is not None and weights_path is not None:
      with open(model_path) as f:
        json_contents = f.read()
      self.model = model_from_yaml(json_contents)
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
    self.camera_to_ground_arc = np.arctan(self.camera_min_view / self.camera_height)
    self.camera_offset_y = 0
    self.camera_arc_y = 80 * (np.pi / 180)
    self.camera_arc_x = 60 * (np.pi / 180)
    self.crop_ratio = [c / s for c, s in zip(self.crop_size, self.source_size)]

      
      
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
              
  #Runs clonemodel
  #seriously Karol, you couldn't keep the variables consistant comming in and out of this function?
  def evaluate(self, frame, speed, steer):
    """ 
    Cut out the patch and run the model on it
    """
    batch = self.preprocess(frame)
    nn_speed, nn_steer = self.model.predict_on_batch(batch)[0]
    return nn_speed, nn_steer, batch[0]

  #runs line_model returns camera pov bezier control points
  def road_spotter(self, frame):

    batch = self.preprocess(frame)

    road_lines = self.model.predict_on_batch(batch)[0]

    return np.reshape(road_lines, (self.n_lines, self.n_dimensions, self.n_points) )

  #This function uses a transform to map the percieved road onto a 2d plane beneath the car
  def road_mapper(self, frame):

    road_spots = self.road_spotter(frame)

    #road_map = np.zeros(np.shape(road_spots), np.float)

    #First we deal with all the points
    road_map[:, 1, :] = self.camera_height * np.tan(self.camera_to_ground_arc + road_spots[:, 0, :] * self.camera_arc_x/self.crop_ratio[1])
    road_map[:, 0, :] = np.multiply( np.power( ( np.power(self.camera_height, 2) + np.power(road_map[:, 1, :], 2) ), 0.5 ) , np.tan(self.camera_offset_y + (road_spots[:, 1, :]-0.5)*self.camera_arc_y ) )

    return road_map

  # pilot_mk1 is currently only built for low speed operation
  def pilot_mk1(self, frame):

    batch = self.preprocess(frame)

    road_map = self.road_mapper( batch)

    #speed is a constant, for now
    nn_speed = .25

    #steering angle is a function of how far the car believes it is from the center of the road
    #note that this is completely un damped and may become unstable at high speeds
    nn_steer = road_map[1, 0, 0] / (road_map[2, 0, 0] - road_map[0, 0, 0])
    
    # center vector is a measure of what direction the road is pointing
    center_vector = road_map[1, :, 1] - road_map[1, :, 0]
    '''
    road angle compensation
    adjusts stearing angle to account for road curvature
    nn_speed term is a linear scaling factor to kill this component at low speeds
    where position correction should be sufficient without a differential component
    (note that the main issue with high speed operation is lag between camera and wheels)
    '''
    nn_steer = nn_steer + 2 * nn_speed * center_vector[0]/center_vector[1]

    return nn_speed, nn_steer, batch[0]
