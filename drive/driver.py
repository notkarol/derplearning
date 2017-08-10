#!/usr/bin/env/python3

import cv2
import time
import curses
from servo import Servo
from camera import Camera



def main(screen):

    # Initialize servo and camera
    servo = Servo()
    # camera = Camera()
    
    # Prepare screen input
    curses.noecho() # don't display pressed characters to screen
    curses.cbreak() # process key instantly without explicit flushing
    screen.keypad(1) # enable keypad mode so keys are not multibyte escape sequences
    screen.nodelay(1) # nonblocking

    # Print out labels
    screen.addstr(0, 0, "ANGLE")
    screen.addstr(1, 0, "SPEED")
    screen.addstr(2, 0, "FPS")
    screen.addstr(3, 0, "")
    screen.addstr(4, 0, "KEY")
    
    # Main loop
    while True:

        # Handle heyboard
        c = screen.getch()
        if c < 0:
            pass
        elif c == ord('q'):
            break 
        elif c == curses.KEY_LEFT:
            servo.turn_left()           # large left turn
        elif c == curses.KEY_RIGHT:
            servo.turn_right()          # large right turn
        elif c == curses.KEY_SLEFT:
            servo.turn_left(.005)       # small left turn
        elif c == curses.KEY_SRIGHT:
            servo.turn_right(.005)      # small right turn
        elif c == ord('/'):
            servo.turn(0)               # sets turn to zero
        elif c == curses.KEY_UP:
            servo.move_faster()         # moves faster
        elif c == curses.KEY_DOWN:
            servo.move_slower()         # slows down
        elif c == ord('.'):
            servo.move(0)               # stops the vehicle

        # Update the screen with the speed and steering
        if c > 0:
            screen.addstr(0, 7, "%06.3f" % servo.angle)
            screen.addstr(1, 7, "%06.3f" % servo.speed)
            screen.addstr(4, 7, chr(c))

        # Refresh the screen and wait before trying again
        screen.refresh()
        time.sleep(1E-3)

    # Prepare screen output
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()

    # Cleanup sensors
    del servo
    # del camera
    
if __name__ == "__main__":
    curses.wrapper(main)
