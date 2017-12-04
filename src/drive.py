#!/usr/bin/env/python3

import argparse
from derp.state import State
import derp.util

def main(args):

    config = derp.util.load_config(args.config)
    config['state']['speed_offset'] = args.speed
    config['state']['steer_offset'] = args.steer

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
        if args.verbose:
            print("%.3f %3s %sspd %.3f %sstr %.3f" %
                  (state['timestamp'] / 1E6, 'REC' if state['record'] else 'off',
                   'A' if state['auto_speed'] else '!', state['speed'],
                   'A' if state['auto_steer'] else '!', state['steer']), end='\r')

        # Exit
        if state.done():
            print("%.3f Exiting" % state['timestamp'])
            return
            

# Load all the arguments and feed them to the main event loader and loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="print a summarized state of the car")
    parser.add_argument('--speed', type=float, default=0.0, help="steer offset")
    parser.add_argument('--steer', type=float, default=0.0, help="speed offset")
    parser.add_argument("config", nargs='+')
    args = parser.parse_args()
    main(args)
