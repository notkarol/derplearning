#!/usr/bin/env/python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import derp.util
import derp.camera
import derp.joystick
import derp.keyboard
import derp.writer
from multiprocessing import Process
import time


def main():
    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=str, help="Main config path, should include all hardeware")
    parser.add_argument("--brain", type=str, help="optional Software config path")
    args = parser.parse_args()

    config = derp.util.load_config(args.config)
    if args.brain:
        config['brain'] = derp.util.load_config(args.brain)

    processes = {}
    for component_name in sorted(config):
        component_parts = component_name.split("_")
        if "camera" in component_parts:
            func = derp.camera.run
        elif "joystick" in component_parts:
            func = derp.joystick.run
        elif "keyboard" in component_parts:
            func = derp.keyboard.run
        elif "writer" in component_parts:
            func = derp.writer.run
        else:
            continue
        print("Starting", component_name)
        processes[component_name] = Process(target=func, args=(config,))
        processes[component_name].start()
    for component_name in sorted(processes):
        processes[component_name].join()


if __name__ == "__main__":
    main()
