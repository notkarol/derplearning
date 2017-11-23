#!/usr/bin/env python3

from derp.scripts.clone import Clone

class CloneFixSpeed(Clone):

    def __init__(self, hw_config, sw_config, path, nocuda):
        super(CloneFixSpeed, self).__init__(hw_config, sw_config, path, nocuda)


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
