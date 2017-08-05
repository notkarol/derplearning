#!/usr/bin/env/python3

import cv2
import time
import servo
import camera

def test_servo():
    s = servo.Servo()

    # Turn right
    s.turn(0.2)
    time.sleep(1)

    # Turn left
    s.turn(-0.2)
    time.sleep(1)

    # Reset forward
    s.turn(0.0)
    time.sleep(1)

    # Move forward
    s.move(0.2)
    time.sleep(0.5)

    # Stop movign
    s.move(0.0)


def test_camera(video_index=1, count=0):
    """
    Show count frames from camera video_index to the screen
    """
    c = camera.Camera(video_index)
    while count >= 0:
        frame = c.getFrame()
        if frame is None:
            print("Skipping frame due to none read")
        cv2.imshow('frame', frame)
    cv2.destroyAllWindows()


def main():
    test_servo()
    test_camera()
    
if __name__ == "__main__":
    main()
