#!/usr/bin/env python3

from derp.scripts.cameraclone import CameraClone

class CameraCloneFixSpeed(CameraClone):

    def __init__(self, sw_config, target_hw_config, source_hw_config, path, nocuda):
        super(CameraCloneFixSpeed, self).__init__(sw_config, target_hw_config,
                                                  source_hw_config, path, nocuda)


    def plan(self, state):
        # Do not do anything if we do not have a loaded model
        if self.model is None:
            return 0.0, 0.0

        # Get the predictions of our model
        predictions = self.predict(state)

        # Prepare params variable for verbosity
        params = self.sw_config['params']
                
        # Figure out future_steer based on various normalizations and weights
        if params['use_min_for_speed']:
            future_steer = float(min(predictions))
        else:
            future_steer = float(predictions[1])
        multiplier = 1 + params['scale'] * (1 - abs(future_steer)) ** params['power']
        speed = state['speed_offset'] * multiplier
        speed = params['speed_curr'] * speed + params['speed_prev'] * self.prev_speed
        self.prev_speed = speed

        # Our steering output
        steer = float(predictions[0])
        steer = params['steer_curr'] * steer + params['steer_prev'] * self.prev_steer
        self.prev_steer = steer
        
        return speed, steer
