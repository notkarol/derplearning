#!/usr/bin/env/python3
from derp.state import State
import argparse
import derp.util

def main(args):

    # Prepare configuration and some supplied arguments
    config = derp.util.load_config(args.car)
    if args.model_dir is not None: config['model_dir'] = args.model_dir
    state, components = derp.util.load_components(config)
    print("%.3f Ready" % state['timestamp'])        

    # Event loop
    while True:

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
            print("\t%.3f %3s %1s spd %.3f %1s str %.3f" %
                  (state['timestamp'] / 1E6, 'REC' if state['record'] else 'off',
                   'A' if state['auto_speed'] else ' ', state['speed'],
                   'A' if state['auto_steer'] else ' ', state['steer']), end='\r')

        # Exit
        if state.done():
            print("%.3f Exiting" % state['timestamp'])
            return
            

# Load all the arguments and feed them to the main event loader and loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--car', type=str, default=derp.util.get_default_config_path(),
                        help="car we are running") 
    parser.add_argument('--model_dir', type=str, default=None,
                        help="directory to models we wish to run")
    parser.add_argument('--quiet', action='store_true', default=False,
                        help="print a summarized state of the car")
    args = parser.parse_args()
    main(args)
