# A class that is used to communicate keyboard, joystick or controller input with the drive code

class Command:

    def __init__(self):
        self.reset()

    def reset(self):
        self.speed = 0
        self.steer = 0
        self.auto_speed = False
        self.auto_steer = False
        self.record = False
        self.alert = False

    def __str__(self):
        return ("Speed %.2f Steer %.2f AutoSpeed %s AutoSteer %s Record %s Alert %s" %
                (self.speed, self.steer,
                 self.auto_speed, self.auto_speed,
                 self.record, self.alert))
