#!/usr/bin/env python3
"""
The main driver program that loops the live internal state.
"""
import argparse
from pathlib import Path
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
    parser.add_argument("config", type=Path, help="Main config path, should include all hardeware")
    args = parser.parse_args()

    config = derp.util.load_config(args.config)

    processes = {}
    for component_name in sorted(config):
        if "camera" == component_name:
            func = derp.camera.run
        elif "input" == component_name:
            if config[component_name]['type'] == 'keyboard':
                func = derp.keyboard.run
            elif config[component_name]['type'] == 'joystick':
                func = derp.joystick.run
            else:
                continue
        elif "writer" == component_name:
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
