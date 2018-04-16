#!/usr/bin/env/python3
import argparse
import os
from time import time
from derp.state import State
import derp.util

def main(args):

    # Prepare configuration and some supplied arguments
    config_path = os.path.join(os.environ['DERP_ROOT'], 'config', args.config + '.yaml')
    config = derp.util.load_config(config_path)
    if args.model_dir is not None:
        config['model_dir'] = args.model_dir
    state, components = derp.util.load_components(config)
    print("%.3f Ready" % state['timestamp'])        

    # Event loop that runs until state is done
    loop_time = time()
    fps = 0
    while not state.done():
        
        # Sense Plan Act Record loop
        for component in components:
            component.sense(state)
        for component in components:
            component.plan(state)
        for component in components:
            component.act(state)
        for component in components:
            component.record(state)
            
        # Print to the screen for verbose mode
        if not args.quiet:
            fps = fps * 0.8 + (1. / (time() - loop_time)) * 0.2
            loop_time = time()
            print("%.3f %2i %s %s | speed %6.3f + %6.3f %i | steer %6.3f + %6.3f %i" %
                  (state['timestamp'], fps + 0.5,
                   'R' if state['record'] else '_',
                   'A' if state['auto'] else '_',
                   state['speed'], state['offset_speed'], state['use_offset_speed'],
                   state['steer'], state['offset_steer'], state['use_offset_steer']))
                                              

# Load all the arguments and feed them to the main event loader and loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=derp.util.get_hostname(),
                        help="physical config")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="directory to models we wish to run")
    parser.add_argument('--quiet', action='store_true', default=False,
                        help="do not print speed/steer")
    args = parser.parse_args()
    main(args)
