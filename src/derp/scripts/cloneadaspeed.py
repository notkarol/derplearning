#!/usr/bin/env python3

from derp.scripts.clone import Clone

class CloneFixSpeed(Clone):

    def __init__(self, config, full_config):
        super(CloneFixSpeed, self).__init__(config, full_config)


    def plan(self, state):
        # Do not do anything if we do not have a loaded model
        if self.model is None:
            return 0.0, 0.0

        # Get the predictions of our model
        predictions = self.predict(state)

        # Figure out future_steer based on various normalizations and weights
        if self.config['use_min_for_speed']:
            future_steer = float(min(predictions))
        else:
            future_steer = float(predictions[1])
        multiplier = 1 + self.config['scale'] * (1 - abs(future_steer)) ** self.config['power']
        speed = state['offset_speed'] * multiplier

        # Our steering output
        steer = float(predictions[0])
        
        return speed, steer
