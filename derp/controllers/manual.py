#!/usr/bin/env python3

from derp.controller import Controller

class Manual(Controller):
    def __init__(self, config, car_config, state):
        super(Manual, self).__init__(config, car_config, state)
        self.ready = True
