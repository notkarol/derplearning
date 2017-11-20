#!/usr/bin/env/python3

import argparse
from derp.inferer import Inferer
from derp.state import State
import derp.util

def main(args):

    # Load hw config config
    hw_config = derp.util.load_config(args.hw)
    
    # Prepare variables
    state = State(args.speed, args.steer)
    components = derp.util.load_components(hw_config, state)
    inferer = Inferer(hw_config, path=args.model)

    # Event loop
    print("Ready")
    while True:

        # Sense Plan Act loop
        for component in components:
            component.sense(state)
        inferer.plan(state)
        for component in components:
            component.act(state)

        # Write out state and each component buffer
        state.scribe(args.hw)
        for component in components:
            component.scribe(state)

        # Print to the screen for verbose mode
        if args.verbose:
            print("%.3f %3s %sspd %.3f %sstr %.3f" %
                  (state['timestamp'] / 1E6, 'REC' if state['record'] else 'off',
                   'A' if state['auto_speed'] else '!', state['speed'],
                   'A' if state['auto_steer'] else '!', state['steer']))

        # Exit
        if state['exit']:
            print("Exiting")
            return
            

# Load all the arguments and feed them to the main event loader and loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw', type=str, required=True,
                        help="path to physical configuration of the car")
    parser.add_argument('--model', type=str, default=None,
                        help="folder where models and configs are stored")
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="print a summarized state of the car")
    parser.add_argument('--speed', type=float, default=0.0, help="steer offset")
    parser.add_argument('--steer', type=float, default=0.0, help="speed offset")
    args = parser.parse_args()    
    main(args)
