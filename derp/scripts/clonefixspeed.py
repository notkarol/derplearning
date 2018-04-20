#!/usr/bin/env python3

from derp.scripts.clone import Clone

class CloneFixSpeed(Clone):

    def __init__(self, config, full_config):
        super(CloneFixSpeed, self).__init__(config, full_config)


    def plan(self, state):
        predictions = self.predict(state)
        speed = state['offset_speed']
        steer = float(predictions[0])
        return speed, steer
