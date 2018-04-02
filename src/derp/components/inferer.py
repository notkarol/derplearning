#!/usr/bin/env python3

import os
from time import time
from derp.component import Component
import derp.util

class Inferer(Component):
    
    def __init__(self, config, full_config):
        super(Inferer, self).__init__(config, full_config)  
        
        # If we have a blank config or path, then assume we can't plan, 
        if 'path' not in config or not config['path']:
            self.script = None
            self.ready = True
            return

        # If we are not given a path then we have no script, and therefore cannot plan
        script_path = 'derp.scripts.%s' % (config['script'].lower())
        script_class = derp.util.load_class(script_path, config['script'])
        self.script = script_class(config, full_config)
        self.ready = True


    def sense(self, state):
        if self.script is None or not state['auto']:
            return True
        return self.script.sense(state)


    def plan(self, state):
        """
        Runs the loaded python inferer script's plan
        """

        # Skip if we have no script to run or we're not asked to control the cor
        if self.script is None or not state['auto']:
            return True

        # Get the proposed list of changes
        speed, steer = self.script.plan(state)

        # Make sure we have the permissions to update these fields
        if state['auto']:
            state['speed'] = speed
            state['steer'] = steer

        return True


    def act(self, state):
        if self.script is None or not state['auto']:
            return True
        return self.script.act(state)


    def record(self, state):
        if self.script is None or not state['auto']:
            return True
        return self.script.record(state)
    
