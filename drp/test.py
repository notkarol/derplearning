import servo
import time

def main():
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

if __name__ == "__main__":
    main()
