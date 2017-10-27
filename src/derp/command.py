# A class that is used to communicate keyboard, joystick or controller input with the drive code

class Command:

    def __init__(self):
        self.reset()

    def reset(self):
        self.steer = 0
        self.speed = 0
        self.auto_steer = False
        self.auto_speed = False
        self.record = False
        self.alert = False

