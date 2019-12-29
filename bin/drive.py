#!/usr/bin/env python3
"""
The main driver program that loops the live internal state.
"""
import argparse
from pathlib import Path
import derp.util
import derp.brain
import derp.camera
import derp.joystick
import derp.writer
from multiprocessing import Process
import time


def all_running(processes):
    for proc in processes:
        proc.join(timeout=0)
        if not proc.is_alive():
            return False
    return True


def main():
    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=Path, help="Main config path, should include all hardeware")
    args = parser.parse_args()

    config = derp.util.load_config(args.config)

    component_map = {'brain': derp.brain.run,
                     'camera': derp.camera.run,
                     'joystick': derp.joystick.run,
                     'writer': derp.writer.run}
    processes = []
    for name in sorted(component_map):
        print('Starting', name)
        proc = Process(target=component_map[name], args=(config,))
        proc.start()
        processes.append(proc)
    while all_running(processes):
        time.sleep(1)
    for proc in processes:
        proc.terminate()


if __name__ == "__main__":
    main()
