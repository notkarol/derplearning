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
    
    # Main loop
    while True:

        # Handle heyboard
        c = screen.getch()
        if c < 0:
            pass
        elif c == ord('q'):
            break 
        elif c == curses.KEY_LEFT:
            servo.turn_left()           #large left turn
            screen.addstr(0, 6, "%06.3f" % servo.angle)
        elif c == curses.KEY_RIGHT:
            servo.turn_right()          #large right turn
            screen.addstr(0, 6, "%06.3f" % servo.angle)
        elif c == curses.KEY_SLEFT:
            servo.turn_left(.005)       #small left turn
            screen.addstr(0, 6, "%06.3f" % servo.angle)
        elif c == curses.KEY_SRIGHT:
            servo.turn_right(.005)      #small right turn
            screen.addstr(0, 6, "%06.3f" % servo.angle)
        elif c == ord('/'):
            servo.turn_zero()           #sets turn to zero
            screen.addstr(0, 6, "%06.3f" % servo.angle)
        elif c == curses.KEY_UP:
            servo.move_faster()         #moves faster
            screen.addstr(1, 6, "%06.3f" % servo.speed)
        elif c == curses.KEY_DOWN:
            servo.move_slower()         #slows down
            screen.addstr(1, 6, "%06.3f" % servo.speed)
        elif c == ord('.'):
            servo.move_zero()           #stops the vehicle
            screen.addstr(1, 6, "%06.3f" % servo.speed)
        else:
            screen.addstr(4, 0, chr(c))
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
