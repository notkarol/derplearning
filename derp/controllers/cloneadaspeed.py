#!/usr/bin/env python3

from derp.scripts.clone import Clone

class CloneAdaSpeed(Clone):

    def __init__(self, config, full_config, state):
        super(CloneAdaSpeed, self).__init__(config, full_config, state)

    def plan(self):
        predictions = self.predict()

        # Future steering angle magnitude dictates speed
        if self.config['use_min_for_speed']:
            future_steer = float(min(predictions))
        else:
            future_steer = float(predictions[1])
        multiplier = 1 + self.config['scale'] * (1 - abs(future_steer)) ** self.config['power']
        speed = self.state['offset_speed'] * multiplier

        steer = float(predictions[0])
        return speed, steer
