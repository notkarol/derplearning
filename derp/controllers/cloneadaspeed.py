#!/usr/bin/env python3

from derp.scripts.clone import Clone

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
