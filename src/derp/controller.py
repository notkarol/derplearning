import asyncio
from datetime import datetime
import evdev
import numpy as np

class Controller:

    def __init__(self, command):
        self.command = command
        
        # Initialize known device names and their handles
        self.devices = {}
        self.settings = {}
        self.initializers = {'Wireless Controller' : self.init_ds4}

        # Connect to new devices
        self.discover()


    def discover(self):
        """
        Find and initialize the available devices
        """
        potential_devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
        for device in potential_devices:

            # Skip devices we're already connected to
            if device in self.devices:
                continue

            # If this device is supported, initialize it
            if device.name in self.initializers:
                self.devices[device] = self.initializers[device.name](device)
                asyncio.ensure_future(self.devices[device](device))

                
    def init_ds4(self, device):
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
        self.ds4_trigger_deadzone = 4

        self.settings[device] = {}
        self.settings[device][self.ds4_left_trigger] = False
        self.settings[device][self.ds4_right_trigger] = False
        self.settings[device][self.ds4_left_stick_horizontal] = False
        self.settings[device][self.ds4_right_stick_horizontal] = False
        
        return self.process_ds4

    def in_deadzone(self, value, deadzone):
        return 128 - deadzone < event.value <= 128 + deadzone


    def normalize_stick(self, value, deadzone):
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value

    
    def normalize_trigger(self, value, deadzone):
        value -= deadzone
        value /= 255 - deadzone
        return value

    
    async def process_ds4(self, device):
        async for event in device.async_read_loop():

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

            # Fixed speed modifications using arrows
            if event.code == self.ds4_arrow_horizontal:
                self.command.speed += 0.01 * event.value
            if event.code == self.ds4_arrow_vertical:
                self.command.speed += 0.1 * event.value
                
            # Handle steer
            if event.code in [self.ds4_left_stick_horizontal, self.ds4_right_stick_horizontal]:

                # Skip if junk
                if event.value == 0:
                    continue
                
                # If it's in the deadzone and it wasn't last time then clear steer
                if self.in_deadzone(event.value, self.ds4_stick_deadzone):
                    if self.settings[event.code]:
                        self.command.steer = 0
                        self.settings[event.code] = False
                    continue
                self.settings[event.code] = True

                # Otherwise use proportional control
                self.command.steer = self.normalize_stick(event.value, self.ds4_stick_deadzone)
                if event.code == self.ds4_right_stick_horizontal:
                    sign = (1 if self.command.steer >= 0 else -1)
                    self.command.steer = sign * (self.command.steer ** 2)
                
            # Handle speed
            if event.code in [self.ds4_left_trigger, self.ds4_right_trigger]:

                # Skip if junk
                if event.value == 0:
                    continue

                # If it's in the deadzone and it wasn't last time then clear speed 
                if self.in_deadzone(event.value, self.ds4_trigger_deadzone):
                    if self.settings[event.code]:
                        self.command.speed = 0
                        self.settings[event.code] = False
                    continue
                self.settings[event.code] = True
                      
                # Otherwise use proportional control
                self.command.speed = self.normalize_trigger(event.value, self.ds4_trigger_deadzone)
                if event.code == self.ds4_right_trigger:
                    sign = (1 if self.command.speed >= 0 else -1)
                    self.command.speed = sign * (self.command.speed ** 2)


