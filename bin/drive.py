#!/usr/bin/env python3
"""
The main driver program that loops the live internal state.
"""
import argparse
import logging
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
    """ Returns whether all processes are currently alive """
    for proc in processes:
        proc.join(timeout=0)
        if not proc.is_alive():
            return False
    return True


def loop(config, exit_event, func):
    """ Makes running multiprocessing easier """
    obj = func(config)
    while not exit_event.is_set() and obj.run():
        pass
    del obj


def main():
    """ Prepare arguments, configurations, variables and run the event loop. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=Path, help="Main config path, should include all hardeware")
    args = parser.parse_args()

    pid_path = '/tmp/derp_drive.pid'
    if derp.util.is_already_running(pid_path):
        return
    derp.util.write_pid(pid_path)

    config = derp.util.load_config(args.config)
    recording_path = derp.util.make_recording_path()
    derp.util.dump_config(config, recording_path / 'config.yaml')
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
        proc = Process(target=loop, name=name, args=proc_args)
        proc.start()
        processes.append(proc)
        logger.info("start %s %i", name, proc.pid)
    while all_running(processes):
        time.sleep(0.1)
    exit_event.set()
    logger.info("exit")

if __name__ == "__main__":
    main()
