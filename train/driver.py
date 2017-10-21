#!/usr/bin/env/python3

import argparse
import curses
from time import sleep, time, strftime, gmtime
from socket import gethostname
import os
from shutil import copyfile
import derputil

# Local Python Files
from camera import Camera
from servo import Servo
from model import Model

def main(screen, args):
    
    # Pepare variables for recording
    date = strftime('%Y%m%dT%H%M%SZ', gmtime())
    name = '%s-%s' % (date, gethostname())
    folder = os.path.join(os.environ['DERP_DATA'], name)
    os.mkdir(folder)
    state_fp = open(os.path.join(folder, "state.csv"), 'w')
    state_fp.write("timestamp,speed,steer\n")
    copyfile(args.config, os.path.join(folder, 'config.yaml'))
    mode = 'record' if args.record else 'drive'
    
    # Initialize relevant classes
    timestamp = int(time() * 1E6)
    config = derputil.loadConfig(args.config)
    camera = Camera(config, folder, mode)
    servo = Servo()
    model = Model(config, folder, mode, args.model) if args.model else None
    autonomous = False

    # Prepare screen input
    curses.noecho() # don't display pressed characters to screen
    curses.cbreak() # process key instantly without explicit flushing
    screen.keypad(1) # enable keypad mode so keys are not multibyte escape sequences
    screen.nodelay(1) # nonblocking

    # Print out labels
    screen.addstr(0, 0, "SPEED")
    screen.addstr(1, 0, "STEER")
    screen.addstr(2, 0, "FPS")
    if args.model: 
        screen.addstr(3, 0, "AUTO")
        screen.addstr(4, 0, "NNSPEED")
        screen.addstr(5, 0, "NNSTEER")
    screen.addstr(6, 0, name)

    # Main loop
    while True:

        last_timestamp = timestamp
        timestamp = int(time() * 1E6)
        frame = camera.getFrame()
        speed, steer = servo.speed, servo.steer
        if autonomous:
            nn_speed, nn_steer, nn_thumb = model.evaluate(frame, timestamp, speed, steer)
            servo.move(0.5 * servo.speed + 0.5 * nn_speed) # dampen
            servo.turn(0.5 * servo.steer + 0.5 * nn_steer) # dampen
            screen.addstr(4, 8, "%6.3f" % nn_speed)
            screen.addstr(5, 8, "%6.3f" % nn_steer)            
        state_fp.write(",".join([str(x) for x in (timestamp, speed, steer)]))
        state_fp.write("\n")
        camera.record()
            
        # Handle heyboard
        c = screen.getch()
        if c < 0:                    pass
        elif c == ord('q'):          break 
        elif c == ord('a'):          autonomous = args.model
        elif c == ord('s'):          autonomous = False ; servo.turn(0) ; servo.move(0)
        elif c == curses.KEY_LEFT:   servo.turn_left(0.2)  # large left turn
        elif c == curses.KEY_RIGHT:  servo.turn_right(0.2) # large right turn
        elif c == curses.KEY_SLEFT:  servo.turn_left(0.05)  # small left turn
        elif c == curses.KEY_SRIGHT: servo.turn_right(0.05) # small right turn
        elif c == curses.KEY_UP:     servo.move_faster()    # moves faster
        elif c == curses.KEY_DOWN:   servo.move_slower()    # slows down
        elif c == ord('0'):          servo.move(0)          # stops the vehicle
        elif c == ord('1'):          servo.move(0.14)       # slowest speed to move
        elif c == ord('2'):          servo.move(0.15)       # moderate speed
        elif c == ord('3'):          servo.move(0.20)       # fastest safe speed
        elif c == ord('/'):          servo.turn(0)          # sets turn to zero

        # Refresh the screen and wait before trying again
        screen.addstr(0, 8, "%6.3f" % servo.speed)
        screen.addstr(1, 8, "%6.3f" % servo.steer)
        screen.addstr(2, 8, "%6.1f" % (1.0 / (timestamp - last_timestamp)))
        if args.model:
            screen.addstr(3, 8, "%6s" % autonomous)
        screen.refresh()
        if not autonomous:
            sleep(1E-2)
        
    # Prepare screen output
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

    # Cleanup sensors
    if args.model:
        del model
    del servo
    del camera
    state_fp.close()
    
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    default_config_path = os.path.join(os.environ['DERP_CONFIG'], 'paras.yaml')
    parser.add_argument('--config', default=default_config_path, help="Configuration to use")
    parser.add_argument('--model', default="", help="Model to run")
    parser.add_argument('--record', action='store_true', help="Whether to run at hi fidelity")
    args = parser.parse_args()

    curses.wrapper(main, args)
