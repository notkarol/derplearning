#!/usr/bin/env python3

import os
import json
from time import time
import derp.util

class Inferer:
    
    def __init__(self, hw_config, sw_config=None, path=None, nocuda=False):
        """
        Loads the supplied python script as this inferer.
        """
        # If we have a blank config or path, then assume we can't plan, 
        if path is None and sw_config is None:
            raise ValueError("Both path and sw_config can not be none")
            return

        # Make sure we have
        if hw_config is None:
            raise ValueError("hw_config can not be none")
        
        # If we are not given a sw config, use the one in path
        if sw_config is None:
            sw_config_path = os.path.join(path, 'sw_config.yaml')
            sw_config = derp.util.load_config(sw_config_path)

        # If we are not given a path then we have no script, and therefore cannot plan
        script_path = 'derp.scripts.%s' % (sw_config['script'].lower())
        script_class = derp.util.load_class(script_path, sw_config['script'])
        self.script = script_class(hw_config, sw_config, path, nocuda)


    def plan(self, state):
        """
        Runs the loaded python inferer script's plan
        """

        # Skip if we're not autonomous
        if not state['auto_speed'] and not state['auto_steer']:
            return True

        # If we have a blank script, drop out
        if self.script is None:
            return True

        # Get the proposed list of changes
        speed, steer = self.script.plan(state)

        # Make sure we have the permissions to update these fields
        if state['auto_speed']:
            state['speed'] = speed
        if state['auto_steer']:
            state['steer'] = steer

        return True


