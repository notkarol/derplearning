#!/usr/bin/env python3

import derp.util as util

class Inferer:
    
    def __init__(self, hw_config, sw_config, model_dir, state):
        """
        Loads the supplied python script as this inferer.
        """

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
        
        # Get the proposed list of changes
        proposal = self.script.plan(state)

        # Make sure we have the permissions to update these fields
        for field, val in proposal:
            var = "auto_%s" % field
            if var in state and not state[var]:
                continue
            state[field] = val

        return True
