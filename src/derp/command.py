# A class that is used to communicate keyboard, joystick or controller input with the drive code

class Command:
    def __init__(self):
        self.time = None
        self.steer = None
        self.speed = None
        self.speed_delta = None
        self.steer_delta = None
        self.auto = None
        self.record = None
        self.stop = None
