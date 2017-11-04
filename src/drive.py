#!/usr/bin/env/python3

import argparse
import derp.component
import derp.infer
import derp.recorder
import derp.state
import derp.util

def main(args):

    # Initialize Configuration
    config = util.loadConfig(args.config)

    # Common state between actuators, sensors, and infer
    state = derp.state.State(config)

    # Prepare the hardware components
    components = util.loadComponents(config)

    # Prepare the inference pipeline
    infer = derp.infer.Infer(config, args.infer)
    
    # Event loop
    while True:
        # Update each component
        for component in components:
            component.sense(state)

        # Plan action
        infer.plan(state)

        # Act upon those actions
        for component in components:
            component.act(state)

        # Store record
        infer.write(state)
        for component in components:
            component.write(state)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="physical configuration")
    parser.add_argument('--infer', type=str, default='', help="infer configuration")
    args = parser.parse_args()
    
    main(args)
