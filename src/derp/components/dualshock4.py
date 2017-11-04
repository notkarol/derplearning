#!/usr/bin/env python3

import evdev
from time import time
from derp.component import Component
import derp.util as util

class Dualshock4(derp.component.Component):

    def __init__(self, config, name, index=None):
        super(Dualshock4).__init__()
        self.command = command
        
        self.device = None

        # Prepare key code
        self.left_stick_horizontal = 0
        self.left_stick_vertical = 1
        self.right_stick_horizontal = 2
        self.left_trigger = 3
        self.right_trigger = 4
        self.right_stick_vertical = 5
        self.arrow_horizontal = 16
        self.arrow_vertical = 17
        self.square = 304
        self.cross = 305
        self.circle = 306
        self.triangle = 307
        self.l1 = 308
        self.r1 = 309
        self.l2 = 310
        self.r2 = 311
        self.share = 312
        self.options = 313
        self.left_stick_press = 314
        self.right_stick_press = 315
        self.menu = 316
        self.touchpad = 317

        # Set an analog stick deadzone
        if 'deadzone' in self.config[name]:
            self.deadzone = self.config[name]['deadzone']
        else:
            self.deadzone = 8

        # Set the speed multipler
        if 'speed_mult' in self.config[name]:
            self.speed_mult = self.config[name]['speed_mult']
        else:
            self.speed_mult = 0.375

        # Set the speed multipler
        if 'speed_pow' in self.config[name]:
            self.speed_pow = self.config[name]['speed_pow']
        else:
            self.speed_pow = 1

        # Set the steer multipler
        if 'steer_mult' in self.config[name]:
            self.steer_mult = self.config[name]['steer_mult']
        else:
            self.steer_mult = 0.375

        # Set the steer multipler
        if 'steer_pow' in self.config[name]:
            self.steer_pow = self.config[name]['steer_pow']
        else:
            self.steer_pow = 1

        self.out_buffer = []
        self.settings = {}
        self.settings[self.left_stick_horizontal] = False
        self.settings[self.left_stick_vertical] = False
        self.settings[self.left_trigger] = 0
        self.settings[self.right_stick_horizontal] = False
        self.settings[self.right_stick_vertical] = False
        self.settings[self.right_trigger] = 0

    def __del__(self):
        if self.device is not None:
            self.device.close()
            self.device = None
        if self.out_csv is not None:
            self.out.csv.close()
            self.out_csv = None
        
    def in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self.deadzone < value <= 128 + self.deadzone


    def act(self, state):
        return True

    
    def discover(self):
        """
        Find and initialize the available devices
        """
        self.device = util.find_device()
        return self.device is not None

    
    def folder(self, folder):
        if self.out_csv is not None:
            self.out.csv.close()
        self.out_csv_path = os.path.join(folder, "%s.csv" % self.name)
        self.out_csv = open(self.name, 'w')
            

    def __normalize_stick(self, value, deadzone):
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value

    
    def sense(self, state):
        try:
            for event in self.device.read():

                # Stop car in every way
                if event.code == self.triangle:
                    continue

                # Sound the alarm!
                if event.code == self.touchpad:
                    state.alert != state.alert

                # Enable autonomous driving
                if event.code == self.options:
                    state.record = True
                    state.auto_steer = True
                if event.code == self.share:
                    state.record = True
                    state.auto_speed = True
                if event.code == self.menu:
                    state.record = True
                    state.auto_steer = True
                    state.auto_speed = True

                # Disable Autonomous
                if event.code == self.cross:
                    state.auto_steer = False
                    state.auto_speed = False

                # Enable/Disable Recording
                if event.code == self.square:
                    state.record = True
                if event.code == self.circle:
                    state.record = False

                # Change wheel offset
                if event.code == self.arrow_horizontal:
                    state.steer_offset += 0.01 * event.value

                # Fixed speed modifications using arrows
                if event.code == self.arrow_vertical:
                    state.speed -= 0.02 * event.value

                # Handle steer
                if event.code in [self.left_stick_horizontal, self.right_stick_horizontal]:

                    # Skip if junk
                    if event.value == 0:
                        continue

                    # If it's in the deadzone and it wasn't last time then clear steer
                    if self.in_deadzone(event.value):
                        if self.settings[filename][event.code]:
                            state.steer = 0
                            self.settings[filename][event.code] = False
                        continue
                    self.settings[filename][event.code] = True

                    # Otherwise use proportional control
                    state.steer = self.__normalize_stick(event.value, self.stick_deadzone)
                    state.steer = (np.sign(state.steer)
                                   * abs(state.steer * self.steer_multiplier)
                                   ** self.steer_power)

                # Handle speed
                if ((event.code == self.l2 and event.value == 0) or
                    (self.settings[filename][self.left_trigger] and
                     self.settings[filename][self.left_trigger] - time() > 0.1)):
                    self.settings[filename][self.left_trigger] = 0
                    state.speed = 0
                if ((event.code == self.l2 and event.value == 0) or
                    (self.settings[filename][self.right_trigger] and
                     self.settings[filename][self.right_trigger] - time() > 0.1)):
                    self.settings[filename][self.right_trigger] = 0
                    state.speed = 0                
                if event.code in [self.left_trigger, self.right_trigger]:
                    if self.in_trigger_deadzone(event.value):
                        continue
                    if event.type == 4:
                        continue
                    self.settings[filename][event.code] = time()
                    state.speed = event.value / 256
                    state.speed = (np.sign(state.speed)
                                   * abs(state.speed * self.speed_multiplier)
                                   ** self.speed_power)
                    if event.code == self.right_trigger:
                        state.speed *= -1
                self.out_buffer.append((int(event.timestamp() * 1E6),
                                        event.type,
                                        event.code,
                                        event.value))
        except BlockingIOError:
            pass
        return True

    
    def write(self):

        if self.out_csv_fp is None:
            return False

        for row in self.out_buffer:
            self.out_csv_fp.write(",".join(row) + "\n")
        self.out_csv_fp.flush()
        self.out_buffer = []
        
        return True    
