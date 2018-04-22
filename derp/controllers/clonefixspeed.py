#!/usr/bin/env python3

from derp.scripts.clone import Clone

class CloneFixSpeed(Clone):

    def __init__(self, config, car_config, state):
        super(CloneFixSpeed, self).__init__(config, car_config, state)

    def plan(self):
        self.predict()
        if not self.state['auto']:
            return        
        self.state['speed'] = self.state['offset_speed']
        self.state['steer'] = float(self.state['prediction'][0])
