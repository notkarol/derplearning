#!/usr/bin/env/python3

import argparse
import curses
from time import sleep, time

# Local Python Files
from camera import Camera
from log import Log
from servo import Servo

def main(screen):

    # Prepare screen input
    curses.noecho() # don't display pressed characters to screen
    curses.cbreak() # process key instantly without explicit flushing
    screen.keypad(1) # enable keypad mode so keys are not multibyte escape sequences
    screen.nodelay(1) # nonblocking

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help="Model to run")
    parser.add_argument('--weights', default="", help="Model to run")
    args = parser.parse_args()

    # Initialize relevant classes
    timestamp = time()
    log = Log()
    camera = Camera(log) if args.model and args.weights else Camera(log, height=1080, width=1920)
    servo = Servo(log)
    if args.model and args.weights:
        from model import Model
        model = Model(log, args.model, args.weights)
    recording = False
    autonomous = False

    # Print out labels
    screen.addstr(0, 0, "SPEED")
    screen.addstr(1, 0, "STEER")
    screen.addstr(2, 0, "FPS")
    screen.addstr(3, 0, "REC")
    screen.addstr(4, 0, "AUTO")
    screen.addstr(5, 0, log.name)

    # Main loop
    while True:

        last_timestamp = timestamp
        timestamp = time()

        if recording or autonomous:
            frame = camera.getFrame()
            speed, steer = servo.speed, servo.steer
            if autonomous:
                # average data
                nn_speed, nn_steer = model.evaluate(frame, speed, steer)

                # dampen
                servo.move(0.5 * servo.speed + 0.5 * nn_speed)
                servo.turn(0.5 * servo.steer + 0.5 * nn_steer)
            else:
                nn_speed, nn_steer = None, None
            log.write((timestamp, speed, nn_speed, steer, nn_steer))
            camera.record(frame)
            
        # Handle heyboard
        c = screen.getch()
        if c < 0:                    pass
        elif c == ord('q'):          break 
        elif c == ord('r'):          recording = True
        elif c == ord('t'):          recording = False
        elif c == ord('a'):          autonomous = args.model and args.weights
        elif c == ord('s'):          autonomous = False
        elif c == curses.KEY_LEFT:   servo.turn_left(1E-2)  # large left turn
        elif c == curses.KEY_RIGHT:  servo.turn_right(1E-2) # large right turn
        elif c == curses.KEY_SLEFT:  servo.turn_left(1E-3)  # small left turn
        elif c == curses.KEY_SRIGHT: servo.turn_right(1E-3) # small right turn
        elif c == curses.KEY_UP:     servo.move_faster()    # moves faster
        elif c == curses.KEY_DOWN:   servo.move_slower()    # slows down
        elif c == ord('0'):          servo.move(0)          # stops the vehicle
        elif c == ord('1'):          servo.move(0.13)       # slowest speed to move
        elif c == ord('2'):          servo.move(0.26)       # moderate speed
        elif c == ord('3'):          servo.move(0.39)       # fastest safe speed
        elif c == ord('/'):          servo.turn(0)          # sets turn to zero

        # Refresh the screen and wait before trying again
        screen.addstr(0, 6, "%6.3f" % servo.speed)
        screen.addstr(1, 6, "%6.3f" % servo.steer)
        screen.addstr(2, 6, "%6.1f" % (1.0 / (timestamp - last_timestamp)))
        screen.addstr(3, 6, "%6s" % recording)
        screen.addstr(4, 6, "%6s" % autonomous)
        screen.refresh()
        if not recording and not autonomous:
            sleep(1E-2)

    # Prepare screen output
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

    # Cleanup sensors
    if args.model and args.weights:
        del model
    del servo
    del camera
    del log
    
if __name__ == "__main__":
    curses.wrapper(main)
