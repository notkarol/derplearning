#!/usr/bin/env python3

import os
import json
from time import time
import derp.util as util

class Inferer:
    
    def __init__(self, hw_config, sw_config, model_dir, state):
        """
        Loads the supplied python script as this inferer.
        """
        self.folder = None
        self.out_json_fp = None
        self.out_buffer = []
        
        # If we have a blank config or model dir, then assume we can't plan 
        if sw_config is None or model_dir is None:
            self.script = None
            return

        script_path = 'derp.scripts.%s' % (sw_config['script'].lower())
        script_class = util.load_class(script_path, sw_config['script'])
        self.script = script_class(hw_config, sw_config, model_dir, state)

    
    def plan(self, state):
        """
        Runs the loaded python inferer script's plan
        """

        # If we have a blank script run that
        if self.script is None:
            return True

        # If we aren't enabled to run either autonomous steer or speed, exit
        if not state['auto_steer'] and not state['auto_speed']:
            return False
        
        # Get the proposed list of changes
        proposal = self.script.plan(state)
        proposal['timestamp'] = int(time() * 1E6)
        self.out_buffer.append(proposal)

        # Make sure we have the permissions to update these fields
        for field in proposal:
            val = proposal[field]
            auto_field = 'auto_%s' % field
            if auto_field in state and state[auto_field]:
                state[field] = float(val)

        return True


    def scribe(self, state):
        if not state['folder'] or state['folder'] == self.folder:
            return False
        self.folder = state['folder']
        
        if self.out_json_fp is not None:
            self.out_json_fp.close()
        self.out_json_path = os.path.join(self.folder, "inferer.json")
        self.out_json_fp = open(self.out_json_path, 'w')
        return True

    
    def write(self):
        if self.out_json_fp is None:
            return False
        for row in self.out_buffer:
            json.dump(self.out_buffer, self.out_json_fp)
            self.out_json_fp.write("\n")
        self.out_json_fp.flush()
        self.out_buffer = []
        return True                 
