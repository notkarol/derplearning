#!/usr/bin/env/python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import time
import derp.state
import derp.util

def loop(state, brain, components):
    """
    The Sense, Plan, Act, Record (SPAR) loop.
    """
    # Reset variables
    state['timestamp'] = time.time()
    state['warn'] = 0

    # Sense Plan Act Record loop where each component runs sequentially.
    for component in components:
        component.sense()
    brain.plan()
    for component in components:
        component.act()
    state.record()

    if state['debug']:
        state.print()
    

def prepare_arguments():
    """
    Prepare the arguments from the user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--car', type=str, default=derp.util.get_hostname(),
                        help='Name of vehicle config with physics. Defaults to host name.')
    parser.add_argument('--brain', type=str, default='cross',
                        help='Name of brain that controls car. Defaults to cross')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Switch on debug model, displaying additional messages')
    args = parser.parse_args()
    return args


def main():
    """
    Prepare ar guments, configurations, variables and tn start the event loop.
    """
    args = prepare_arguments()

    car_config = derp.util.load_config(derp.util.get_car_config_path(args.car))
    brain_config = derp.util.load_config(derp.util.get_brain_config_path(args.brain))
    state = derp.state.State(car_config, brain_config, args.debug)
    components = derp.util.load_components(car_config['components'], state)
    brain = derp.util.load_brain(brain_config, car_config, state)

    # The program's running loop
    while not state.done():
        loop(state, brain, components)


if __name__ == "__main__":
    main()
