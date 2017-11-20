#!/usr/bin/env python3

import os
import json
from time import time
import derp.util as util

class Inferer:
    
    def __init__(self, source_hw_config, sw_config=None, path=None, nocuda=False):
        """
        Loads the supplied python script as this inferer.
        """
        # If we have a blank config or path, then assume we can't plan
        if path is None and sw_config is None:
            self.script = None
            return

        # If we do not have a specified sw_config, load it from the path
        if sw_config is None:
            sw_config_path = os.path.join(path, 'sw_config.yaml')
            sw_config = derp.util.load_config(sw_config_path)

        if path is None:
            target_hw_config = source_hw_config
        else:
            target_hw_config_path = os.path.join(path, 'hw_config.yaml')
            target_hw_config = derp.util.load_config(target_hw_config_path)
        
        # Now that we have a sw config, load everything.
        # If the path is None, then we can not call plan
        script_path = 'derp.scripts.%s' % (sw_config['script'].lower())
        script_class = util.load_class(script_path, sw_config['script'])
        self.script = script_class(sw_config, target_hw_config, sw_config,
                                   source_hw_config, path, nocuda)


    def plan(self, state):
        """
        Runs the loaded python inferer script's plan
        """

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


