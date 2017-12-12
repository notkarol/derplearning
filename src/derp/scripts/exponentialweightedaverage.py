#!/usr/bin/env python3

from derp.component import Component

class ExponentialWeightedAverage(Component):

    def __init__(self, config, full_config):
        super(ExponentialWeightedAverage, self).__init__(config, full_config)
        self.prev_steer = 0
        self.prev_speed = 0 
        

    def plan(self, state):

        # Prepare spee dand steer
        speed = self.config['speed'][0] * state['speed'] + self.config['speed'][1] * self.prev_speed
        steer = self.config['steer'][0] * state['steer'] + self.config['steer'][1] * self.prev_steer

        # Track historical value for weighting
        self.prev_speed = speed
        self.prev_steer = steer
        
        return speed, steer
