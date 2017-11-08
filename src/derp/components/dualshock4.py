#!/usr/bin/env python3

import evdev
import numpy as np
import os
from time import time
from derp.component import Component
import derp.util as util

class Dualshock4(Component):

    def __init__(self, config, name):
        super(Dualshock4, self).__init__(config, name)
        self.device = None
        self.out_csv_fp = None
        
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

        self.deadzone = self.config['deadzone']
        
        
        # store when something was last active for deadman switches
        self.active = {}
        self.active[self.left_stick_horizontal] = None
        self.active[self.left_stick_vertical] = None
        self.active[self.left_trigger] = None
        self.active[self.right_stick_horizontal] = None
        self.active[self.right_stick_vertical] = None
        self.active[self.right_trigger] = None

        
    def __del__(self):
        if self.device is not None:
            self.device.close()
            self.device = None
        if self.out_csv_fp is not None:
            self.out_csv_fp.close()
            self.out_csv_fp = None

            
    def in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self.deadzone < value <= 128 + self.deadzone


    def act(self, state):
        return True

    
    def discover(self):
        """
        Find and initialize the available devices
        """
        self.device = util.find_device('Wireless Controller')
        return self.device is not None

    
    def scribe(self, state):
        if not state['folder'] or state['folder'] == self.folder:
            return False
        self.folder = state['folder']

        #  Open csv writer
        if self.out_csv_fp is not None:
            self.out_csv_fp.close()
        self.out_csv_path = os.path.join(self.folder, "%s.csv" % self.name)
        self.out_csv_fp = open(self.out_csv_path, 'w')
        
        return True

    def __normalize_stick(self, value, deadzone):
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value
    

    def __process(self, state, event):
        speed = None
        steer = None
        record = None
        auto_speed = None
        auto_steer = None
        steer_offset = None
        exit = None

        # skip vertical analog stick movement
        if event.code == self.left_stick_vertical or event.code == self.right_stick_vertical:
            return False

        # Handle steer
        elif event.code == self.left_stick_horizontal or event.code == self.right_stick_horizontal:

            # Skip if junk
            if event.value == 0:
                return False

            # If it's in the deadzone and it wasn't last time then clear steer
            if self.in_deadzone(event.value):
                if self.active[event.code]:
                    steer = 0
                    self.active[event.code] = 0
                else:
                    return False
            # Otherwise handle this event
            else:
                self.active[event.code] = True

                # Otherwise use proportional control
                steer = self.__normalize_stick(event.value, self.deadzone)
                if event.code == self.right_stick_horizontal: 
                    sign = np.sign(steer)
                    steer = abs(steer)
                    steer *= self.config['steer_normalizer'][1]
                    steer **= self.config['steer_normalizer'][2]
                    steer += self.config['steer_normalizer'][0]
                    steer = max(0, min(1, steer))
                    steer *= sign
                    steer = float(steer)

        # Handle speed 
        elif event.code == self.left_trigger or event.code == self.right_trigger:
            if event.type == 4:
                return False
            self.active[event.code] = time()

            # Normalize speed
            speed = event.value / 256

            # Further refine curve of speed resistance
            z, x, y = self.config['speed_elbow']
            if speed > 0:
                if speed < x:
                    speed = z + speed * (y - z) / x
                else:
                    speed = (y + (speed - x) * (1 - y) / (1 - x))
                if event.code == self.right_trigger:
                    speed *= -1

        # Handle speed timeouts
        elif ((event.code == self.l1 and event.value == 0) or
            (self.active[self.left_trigger] and
             self.active[self.left_trigger] - time() > 0.1)): # timeout
            self.active[self.left_trigger] = 0
            speed = 0
        elif ((event.code == self.l2 and event.value == 0) or
            (self.active[self.right_trigger] and
             self.active[self.right_trigger] - time() > 0.1)): # timeout
            self.active[self.right_trigger] = 0
            speed = 0
                        
        # Stop car in every way
        elif event.code == self.triangle:
            speed = 0
            steer = 0
            record = False
            auto_speed = False
            auto_steer = False

        # Start autonomous driving in some way
        elif event.code == self.share:
            if not state['record']:
                record = util.get_record_name()
            auto_speed = True
        elif event.code == self.options:
            if not state['record']:
                record = util.get_record_name()
            auto_steer = True
        elif event.code == self.menu:
            if not state['record']:
                record = util.get_record_name()
            auto_speed = True
            auto_steer = True

        # Disable Autonomous
        elif event.code == self.cross:
            auto_speed = False
            auto_steer = False

        # Enable/Disable Recording
        elif event.code == self.square:
            record = False
        elif event.code == self.circle:
            if not state['record']:
                record = util.get_record_name()

        # Change wheel offset
        elif event.code == self.arrow_horizontal:
            steer_offset = state['steer_offset'] + event.value / 128

        # Fixed speed modifications using arrows
        elif event.code == self.arrow_vertical:
            speed = state['speed'] - 0.015625 * event.value

        # TODO create an event
        elif event.code == self.touchpad:
            speed = 0
            steer = 0
            record = False
            auto_speed = False
            auto_steer = False
            exit = True
            
        # Handle responses
        if speed is not None:
            state['speed'] = speed
        if steer is not None:
            state['steer'] = steer
        if record is not None:
            state['record'] = record
        if auto_speed is not None:
            state['auto_speed'] = auto_speed
        if auto_steer is not None:
            state['auto_steer'] = auto_steer
        if steer_offset is not None:
            state['steer_offset'] = steer_offset
        if exit is not None:
            state['exit'] = exit

        # Store this event
        if state['record']:
            self.out_buffer.append((int(time() * 1E6), int(event.timestamp() * 1E6),
                                    event.type, event.code, event.value,
                                    speed, steer, record, auto_speed, auto_steer, steer_offset))
        return True


    def sense(self, state):
        out = {}
        try:
            for event in self.device.read():
                self.__process(state, event)
        except BlockingIOError:
            pass
        return True

    
    def write(self):

        if self.out_csv_fp is None:
            return False

        for row in self.out_buffer:
            self.out_csv_fp.write(",".join([str(x) for x in row]) + "\n")
        self.out_csv_fp.flush()
        self.out_buffer = []
        
        return True    
