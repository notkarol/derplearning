#!/usr/bin/env python3

import os
from time import time
from derp.component import Component
import derp.util

class Controller(Component):
    
    def __init__(self, config, full_config, state):
        self.config = config
        self.full_config = full_config
        self.state = state
        self.ready = True

    def plan(self):
        return True
