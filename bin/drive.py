#!/usr/bin/env python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import logging
from multiprocessing import Event, Process
from pathlib import Path
import time
import yaml
import derp.util
import derp.brain
import derp.camera
import derp.imu
import derp.joystick
import derp.servo
import derp.writer


def all_running(processes):
    """ Returns whether all processes are currently alive """
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
    recording_path = derp.util.make_recording_path()
    with open(str(recording_path / "config.yaml"), "w") as config_fd:
        yaml.dump(config, config_fd)
    config['recording_path'] = recording_path
    logger = derp.util.init_logger('drive', config['recording_path'])
        
    component_map = {
        "brain": derp.brain.Clone,
        "camera": derp.camera.Camera,
        "imu": derp.imu.Imu,
        "joystick": derp.joystick.Joystick,
        "servo": derp.servo.Servo,
        "writer": derp.writer.Writer,
    }
    processes = []
    exit_event = Event()
    for name in sorted(component_map):
        if name not in config:
            logger.info("skip %s", name)
            continue
        proc_args = (config, exit_event, component_map[name])
        proc = Process(target=derp.util.loop, name=name, args=proc_args)
        proc.start()
        processes.append(proc)
        logger.info("start %s %i", name, proc.pid)
    while all_running(processes):
        time.sleep(0.1)
    exit_event.set()
    logger.info("exit")

if __name__ == "__main__":
    main()
