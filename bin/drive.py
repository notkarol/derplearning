#!/usr/bin/env/python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import time
import derp.state
import derp.util
import logging


def main():
    # define logger file
    logging.basicConfig(
        filename="../data/logs/drive_" + datetime.now(),
        format='%(created)s %(levelname)-8s %(message)s',
        level=logging.INFO
        # datefmt='%Y-%m-%d %H:%M:%S'
    )

    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--car', type=str, default=derp.util.get_hostname(),
                        help='Name of components config')
    parser.add_argument('--brain', type=str, default='cross',help='Name of brain config')
    args = parser.parse_args()

    car_config = derp.util.load_config(derp.util.get_car_config_path(args.car))
    brain_config = derp.util.load_config(derp.util.get_brain_config_path(args.brain))

    state = derp.state.State(car_config, brain_config)
    components = derp.util.load_components(car_config['components'], state)
    brain = derp.util.load_brain(brain_config, car_config, state)

    while not state.done():
        state.reset()

        for component in components:
            component.sense()
        logging.info("step: %s sensing complete" % state['timestamp'])

        brain.plan()
        logging.info("step: %s planning complete" % state['timestamp'])

        for component in components:
            component.act()
        logging.info("step: %s acting complete" % state['timestamp'])

        state.record()
        logging.info("step: %s recording complete" % state['timestamp'])
        logging.info(state)

if __name__ == "__main__":
    main()
