#!/usr/bin/env/python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import derp.util
import derp.camera
import derp.keyboard
import derp.writer
from multiprocessing import Process
import time

def main():
    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--car', type=str, required=True, help='Hardware config path')
    parser.add_argument('--brain', type=str, required=True, help='Software config path')
    args = parser.parse_args()

    car_config = derp.util.load_config(args.car)
    brain_config = derp.util.load_config(args.brain)

    processes = {}
    for component_name in sorted(car_config):
        component_parts = component_name.split('_')
        if 'camera' in component_parts:
            func = derp.camera.run
        elif 'keyboard' in component_parts:
            func = derp.keyboard.run
        elif 'writer' in component_parts:
            func = derp.writer.run
        else:
            continue
        print("Starting", component_name)
        processes[component_name] = Process(target=func, args=(car_config[component_name],))
        processes[component_name].start()

    time.sleep(5)
    for component_name in sorted(processes):
        processes[component_name].join()


if __name__ == "__main__":
    main()
