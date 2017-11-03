from datetime import datetime
import evdev
import numpy as np
import time
class Controller:

    def __init__(self, command):
        self.command = command
        
        # Initialize known device names and their handles
        self.devices = {}
        self.settings = {}
        self.runners = {}
        self.initializers = {'Wireless Controller' : self.init_ds4}

        # Connect to new devices
        self.discover()


    def discover(self):
        """
        Find and initialize the available devices
        """
        for filename in evdev.list_devices():

            # Skip devices we're already connected to
            if filename in self.devices:
                continue
            
            # Connect to device
            device = evdev.InputDevice(filename)

            # Skip devices we can't handle
            if device.name not in self.initializers:            
                continue

            self.devices[filename] = device
            self.settings[filename] = {}
            self.runners[filename] = self.initializers[device.name](filename)
                
    def init_ds4(self, filename):
        """
        Prepare necessary variables to process the Dualshock 4 Controller
        """
        self.ds4_left_stick_horizontal = 0
        self.ds4_left_stick_vertical = 1
        self.ds4_right_stick_horizontal = 2
        self.ds4_left_trigger = 3
        self.ds4_right_trigger = 4
        self.ds4_right_stick_vertical = 5
        self.ds4_arrow_horizontal = 16
        self.ds4_arrow_vertical = 17
        self.ds4_square = 304
        self.ds4_cross = 305
        self.ds4_circle = 306
        self.ds4_triangle = 307
        self.ds4_l1 = 308
        self.ds4_r1 = 309
        self.ds4_l2 = 310
        self.ds4_r2 = 311
        self.ds4_share = 312
        self.ds4_options = 313
        self.ds4_left_stick_press = 314
        self.ds4_right_stick_press = 315
        self.ds4_menu = 316
        self.ds4_touchpad = 317

        self.ds4_stick_deadzone = 8
        self.ds4_trigger_deadzone = 1
        self.settings[filename][self.ds4_left_stick_horizontal] = False
        self.settings[filename][self.ds4_left_stick_vertical] = False
        self.settings[filename][self.ds4_left_trigger] = 0
        self.settings[filename][self.ds4_right_stick_horizontal] = False
        self.settings[filename][self.ds4_right_stick_vertical] = False
        self.settings[filename][self.ds4_right_trigger] = 0
        
        return self.process_ds4

    def in_stick_deadzone(self, value, deadzone):
        return 128 - deadzone < value <= 128 + deadzone

    def in_trigger_deadzone(self, value, deadzone):
        return value < deadzone
    

    def normalize_stick(self, value, deadzone):
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value

    
    def normalize_trigger(self, value, deadzone):
        value -= deadzone
        value /= 255 - deadzone
        return value

    
    def process_ds4(self, filename):
        for event in self.devices[filename].read():
            
            # Stop car in every way
            if event.code == self.ds4_triangle:
                self.command.reset()
                continue

            # Sound the alarm!
            if event.code == self.ds4_touchpad:
                self.command.alert != self.command.alert

            # Enable autonomous driving
            if event.code == self.ds4_options:
                self.command.record = True
                self.command.auto_steer = True
            if event.code == self.ds4_share:
                self.command.record = True
                self.command.auto_speed = True
            if event.code == self.ds4_menu:
                self.command.record = True
                self.command.auto_steer = True
                self.command.auto_speed = True

            # Disable Autonomous
            if event.code == self.ds4_cross:
                self.command.auto_steer = False
                self.command.auto_speed = False
                
            # Enable/Disable Recording
            if event.code == self.ds4_square:
                self.command.record = True
            if event.code == self.ds4_circle:
                self.command.record = False

            # Change wheel offset
            if event.code == self.ds4_arrow_horizontal:
                self.command.steer_offset += 0.01 * event.value

            # Fixed speed modifications using arrows
            if event.code == self.ds4_arrow_vertical:
                self.command.speed -= 0.02 * event.value
                
            # Handle steer
            if event.code in [self.ds4_left_stick_horizontal, self.ds4_right_stick_horizontal]:

                # Skip if junk
                if event.value == 0:
                    continue
                
                # If it's in the deadzone and it wasn't last time then clear steer
                if self.in_stick_deadzone(event.value, self.ds4_stick_deadzone):
                    if self.settings[filename][event.code]:
                        self.command.steer = 0
                        self.settings[filename][event.code] = False
                    continue
                self.settings[filename][event.code] = True

                # Otherwise use proportional control
                self.command.steer = self.normalize_stick(event.value, self.ds4_stick_deadzone)
                if event.code == self.ds4_right_stick_horizontal:
                    self.command.steer = self.command.steer ** 3
                
            # Handle speed
            if ((event.code == self.ds4_l2 and event.value == 0) or
                (self.settings[filename][self.ds4_left_trigger] and
                 self.settings[filename][self.ds4_left_trigger] - time.time() > 0.1)):
                self.settings[filename][self.ds4_left_trigger] = 0
                self.command.speed = 0
            if ((event.code == self.ds4_l2 and event.value == 0) or
                (self.settings[filename][self.ds4_right_trigger] and
                 self.settings[filename][self.ds4_right_trigger] - time.time() > 0.1)):
                self.settings[filename][self.ds4_right_trigger] = 0
                self.command.speed = 0                
            if event.code in [self.ds4_left_trigger, self.ds4_right_trigger]:
                if self.in_trigger_deadzone(event.value, self.ds4_trigger_deadzone):
                    continue
                if event.type == 4:
                    continue
                self.settings[filename][event.code] = time.time()
                self.command.speed = self.normalize_trigger(event.value, self.ds4_trigger_deadzone)
                self.command.speed /= 3
                if event.code == self.ds4_right_trigger:
                    self.command.speed *= -1


    def process(self):
        for filename in self.runners:
            try:
                self.runners[filename](filename)
            except BlockingIOError:
                pass
