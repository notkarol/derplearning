#!/usr/bin/env/python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import time
import derp.state
import derp.util

def loop(state, controller, components):
    """
    The Sense, Plan, Act, Record (SPAR) loop.
    """

    # Reset variables
    state['timestamp'] = time.time()
    state['warn'] = 0

    # Sense Plan Act Record loop where each component runs sequentially.
    for component in components:
        component.sense()
    controller.plan()
    for component in components:
        component.act()
    state.record()


def prepare_arguments():
    """
    Prepare the arguments from the user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--car', type=str, default=derp.util.get_hostname(),
                        help="location of config file for vehicle")
    parser.add_argument('--controller', type=str, default='manual',
                        help="location of controller folder")
    parser.add_argument('--quiet', action='store_true', default=False,
                        help="do not print speed/steer")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="don't encapsulate everything in a try-except")
    args = parser.parse_args()
    return args


def main():
    """
    Prepare ar guments, configurations, variables and tn start the event loop.
    """
    args = prepare_arguments()

    # Prepare the configs from the arguments
    car_config_path = derp.util.get_car_config_path(args.car)
    car_config = derp.util.load_config(car_config_path)
    controller_config_path = derp.util.get_controller_config_path(args.controller)
    controller_config = derp.util.load_config(controller_config_path)

    # Prepare the car's major components
    state = derp.state.State(car_config, controller_config)
    components = derp.util.load_components(car_config['components'], state)
    controller = derp.util.load_controller(controller_config, car_config, state)

    # The program's running loop
    while not state.done():
        loop(state, controller, components)
        if not args.quiet:
            state.print()


if __name__ == "__main__":
    main()
