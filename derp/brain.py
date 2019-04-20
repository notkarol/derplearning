"""
The root class of any object that manipulate's the car state based on some heuristic.
"""
import cv2
import numpy as np
import os
import torch
import derp.util

class Brain:
    """
    The root class of any object that manipulate's the car state based on some heuristic.
    """

    def __init__(self, config, car_config, state):
        """
        Preset some common constructor parameters.
        """
        self.config = config
        self.car_config = car_config
        self.state = state
        self.ready = True

    def __repr__(self):
        """
        Unique instances should not really exist so just use the class name.
        """
        return self.__class__.__name__.lower()

    def __str__(self):
        """
        Just use the representation.
        """
        return repr(self)

    def plan(self):
        """
        By default if a child does not override this, do nothing to update the state.
        """
        return True


class Manual(Brain):
    def __init__(self, config, car_config, state):
        super(Manual, self).__init__(config, car_config, state)
        self.ready = True


class Clone(Brain):

    def __init__(self, config, car_config, state):

        super(Clone, self).__init__(config, car_config, state)
        self.camera_config = derp.util.find_component_config(car_config,
                                                             config['thumb']['component'])

        # Show the user what we're working with
        derp.util.print_image_config('Source', self.camera_config)
        derp.util.print_image_config('Target', self.config['thumb'])
        for key in sorted(self.camera_config):
            print("Camera %s: %s" % (key, self.camera_config[key]))
        for key in sorted(self.config['thumb']):
            print("Target %s: %s" % (key, self.config['thumb'][key]))

        
        # Prepare camera inputs
        self.bbox = derp.util.get_patch_bbox(self.config['thumb'], self.camera_config)
        self.size = (config['thumb']['width'], config['thumb']['height'])

        # Prepare model
        self.model_dir = derp.util.get_brain_models_path(self.config['name'])
        self.model_path = derp.util.find_matching_file(self.model_dir, 'clone.pt$')
        if self.model_path is not None and self.model_path.exists():
            self.model = torch.load(str(self.model_path))
            self.model.eval()
        else:
            self.model = None
            print("Clone: Unable to find model path [%s]" % self.model_path)

        # Useful variables for params
        self.prev_steer = 0
        self.prev_speed = 0

        # Data saving
        self.frame_counter = 0  
 
    def prepare_thumb(self, frame):
        if frame is not None:
            patch = derp.util.crop(frame, self.bbox)
            thumb = derp.util.resize(patch, self.size)
            if 'debug' in self.state and self.state['debug']:
                cv2.imshow('patch', patch)
                cv2.waitKey(1)
        else:
            dim = [self.config['thumb']['height'],
                   self.config['thumb']['width']]
            if self.config['thumb']['depth'] > 1:
                dim += [self.config['thumb']['depth']]
            thumb = np.zeros(dim, dtype=np.float32)
        return thumb

    def predict(self):
        status = derp.util.extractList(self.config['status'], self.state)
        frame = self.state[self.config['thumb']['component']]
        self.state['thumb'] = self.prepare_thumb(frame)
        status_batch = derp.util.prepareVectorBatch(status)
        thumb_batch = derp.util.prepareImageBatch(self.state['thumb'])
        status_batch = derp.util.prepareVectorBatch(status)
        if self.model:
            prediction_batch = self.model(thumb_batch, status_batch)
            prediction = derp.util.unbatch(prediction_batch)
            derp.util.unscale(self.config['predict'], prediction)
        else:
            prediction = np.zeros(len(self.config['predict']), dtype=np.float32)
        self.state['prediction'] = prediction
        

    def plan(self):
        self.predict()
        if self.state['auto']:
            self.state['speed'] = float(self.state['prediction'][0])
            self.state['steer'] = float(self.state['prediction'][1])
        

class CloneAdaSpeed(Clone):

    def __init__(self, config, car_config, state):
        super(CloneAdaSpeed, self).__init__(config, car_config, state)

    def plan(self):
        self.predict()
        if self.state['auto']:
            return
    
        # Future steering angle magnitude dictates speed
        if self.config['use_min_for_speed']:
            future_steer = float(min(self.state['prediction']))
        else:
            future_steer = float(self.state['prediction'][1])
        multiplier = 1 + self.config['scale'] * (1 - abs(future_steer)) ** self.config['power']

        self.state['speed'] = self.state['offset_speed'] * multiplier
        self.state['steer'] = float(self.state['predictions'][0])


class CloneFixSpeed(Clone):

    def __init__(self, config, car_config, state):
        super(CloneFixSpeed, self).__init__(config, car_config, state)

    def plan(self):
        self.predict()
        if not self.state['auto']:
            return        
        self.state['speed'] = self.state['offset_speed']
        self.state['steer'] = float(self.state['prediction'][0])
