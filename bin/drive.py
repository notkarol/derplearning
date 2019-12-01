#!/usr/bin/env/python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import derp.util
import derp.camera

def main():
    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--car', type=str, required=True, help='Hardware config path')
    parser.add_argument('--brain', type=str, required=True, help='Software config path')
    args = parser.parse_args()

    car_config = derp.util.load_config(args.car)
    brain_config = derp.util.load_config(args.brain)

    derp.camera.run(car_config)

    

if __name__ == "__main__":
    main()
