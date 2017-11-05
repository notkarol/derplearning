#!/usr/bin/env/python3

import argparse
from derp.inferer import Inferer
from derp.state import State
import derp.util

def main(args):

    config = derp.util.load_config(args.config)
    state = State()
    components = derp.util.load_components(config, state)
    #infer = Inferer(config, args.infer, args.model)
    
    # Event loop
    while True:

        # Sense Plan Act loop
        for component in components:
            component.sense(state)
        # infer.plan(state)
        for component in components:
            component.act(state)

        # Write out state and each component buffer
        if state['record']:
            state.scribe()
            state.write()
            for component in components:
                component.scribe(state)
                component.write()

        # If we're printing do so
        if args.verbose:
            print(repr(state))

        # Exit
        if 'exit' in state and state['exit']:
            return
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="physical configuration")
    parser.add_argument('--infer', type=str, default=None, help="infer configuration")
    parser.add_argument('--model', type=str, default=None, help="model to load")
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    
    main(args)
