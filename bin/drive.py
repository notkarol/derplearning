#!/usr/bin/env python3
"""
The main driver program that loops the live internal state.
"""
import argparse
from multiprocessing import Event, Process
from pathlib import Path
import time
import derp.util
import derp.brain
import derp.camera
import derp.imu
import derp.joystick
import derp.servo
import derp.writer

def all_running(processes):
    for proc in processes:
        proc.join(timeout=0)
        if not proc.is_alive():
            print('Died:', proc.name, proc.pid)
            return False
    return True


def main():
    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=Path, help="Main config path, should include all hardeware")
    args = parser.parse_args()

    config = derp.util.load_config(args.config)
    exit_event = Event()

    component_map = {'brain': derp.brain.Clone,
                     'camera': derp.camera.Camera,
                     'imu': derp.imu.Imu,
                     'joystick': derp.joystick.Joystick,
                     'servo': derp.servo.Servo,
                     'writer': derp.writer.Writer}
    processes = []
    for name in sorted(component_map):
        if name not in config:
            continue
        proc_args = (config, exit_event, component_map[name])
        proc = Process(target=derp.util.loop, name=name, args=proc_args)
        proc.start()
        print('Started:', name, proc.pid)
        processes.append(proc)
    while all_running(processes):
        time.sleep(0.5)
    exit_event.set()


if __name__ == "__main__":
    main()
