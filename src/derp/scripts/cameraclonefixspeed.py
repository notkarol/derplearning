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

        # Prepare parameters variable for verbosity
        params = self.sw_config['params']
        
        # Speed is fixed based on state
        speed = state['speed_offset']

        # Steer is a simple weighted average of the previous speed and the current
        steer = float(predictions[0])
        steer = params['steer_curr'] * steer + params['steer_prev'] * self.prev_steer
        self.prev_steer = steer
        
        return speed, steer
