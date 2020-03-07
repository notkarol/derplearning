"""
Keyboard to drive the car around manually without keyboard.
Inspired by https://github.com/chrippa/ds4drv
"""
import fcntl
from io import FileIO
import os
from struct import Struct
import time
import evdev
from binascii import crc32
from pyudev import Context
from derp.part import Part
import derp.util


class Keyboard(Part):
    """ keyboard to control the car """

    def __init__(self, config):
        """Keyboard to drive the car around manually without keyboard."""
        super(Keyboard, self).__init__(config, "keyboard", [])

        self.code_map = {1: 'escape',
                         2: '1',
                         3: '2',
                         4: '3',
                         5: '4',
                         6: '5',
                         7: '6',
                         8: '7',
                         9: '8',
                         10: '9',
                         11: '0',
                         12: '-_',
                         13: '=+',
                         14: 'backspace',
                         15: 'tab',
                         16: 'q',
                         17: 'w',
                         18: 'e',
                         19: 'r',
                         20: 't',
                         21: 'y',
                         22: 'u',
                         23: 'i',
                         24: 'o',
                         25: 'p',
                         26: '[',
                         27: ']',
                         28: 'enter',
                         29: 'left_ctrl',
                         30: 'a',
                         31: 's',
                         32: 'd',
                         33: 'f',
                         34: 'g',
                         35: 'h',
                         36: 'j',
                         37: 'k',
                         38: 'l',
                         39: ';',
                         40: "'",
                         41: '`',
                         42: 'left_shift',
                         43: '\\',
                         44: 'z',
                         45: 'x',
                         46: 'c',
                         47: 'v',
                         48: 'b',
                         49: 'n',
                         50: 'm',
                         51: ',',
                         52: '.',
                         53: '/',
                         54: 'right_shift',
                         55: 'right_*',
                         56: 'left_alt',
                         57: 'space',
                         58: 'capslock',
                         59: 'f1',
                         60: 'f2',
                         61: 'f3',
                         62: 'f4',
                         63: 'f5',
                         64: 'f6',
                         65: 'f7',
                         66: 'f8',
                         67: 'f9',
                         68: 'f10',
                         69: 'numlock',
                         70: 'scrolllock',
                         71: 'keypad_7',
                         72: 'keypad_8',
                         73: 'keypad_9',
                         74: 'keypad_-',
                         75: 'keypad_4',
                         76: 'keypad_5',
                         77: 'keypad_6',
                         78: 'keypad_+',
                         79: 'keypad_1',
                         80: 'keypad_2',
                         81: 'keypad_3',
                         82: 'keypad_0',
                         83: 'keypad_..',
                         96: 'keypad_enter',
                         97: 'right_ctrl',
                         98: 'keypad_/',
                         100: 'right_alt',
                         102: 'home',
                         103: 'arrow_up',
                         104: 'pagedown',
                         105: 'arrow_left',
                         106: 'arrow_right',
                         107: 'end',
                         108: 'arrow_down',
                         109: 'pagedown',
                         110: 'insert',
                         111: 'delete',
                         125: 'super',
                     }
        # State/Controls
        self.speed = 0
        self.steer = 0
        self.speed_offset = 0
        self.steer_offset = 0
        self.device = self.find_device(self._config['device_names'])
    
    def find_device(self, names):
        """
        Searches for an input devices. Assuming it is found that device is returned
        """
        for filename in sorted(evdev.list_devices()):
            device = evdev.InputDevice(filename)
            device_name = device.name.lower()
            for name in names:
                if name in device_name:
                    print("Using evdev:", device_name)
                    return device
        return None

    def __del__(self):
        self.publish("action", isManual=True, speed=0, steer=0)
        self.publish("controller", isAutonomous=False, speedOffset=0, steerOffset=0, exit=True)
        super(Keyboard, self).__del__()
        if self.device is not None:
            self.device.close()

    def process(self, event):
        print(event)
        # Skip events that I don't know what they mean, but appear all the time
        if event.code == 0 or event.code == 4:
            return

        # Set steer
        self.controller_changed = False
        self.action_changed = False
        if self.code_map[event.code] == 'arrow_left' and event.value:
            self.steer -= 16 / 255
            self.action_changed = True
        if self.code_map[event.code] == 'arrow_right' and event.value:
            self.steer += 16 / 255
            self.action_changed = True

        # Set speed
        if self.code_map[event.code] == 'arrow_up' and event.value:
            self.speed += 4 / 255
            self.action_changed = True
        if self.code_map[event.code] == 'arrow_down' and event.value:
            self.speed -= 4 / 255
            self.action_changed = True
        
        # set steer offset
        if self.code_map[event.code] == '[' and event.value:
            self.steer_offset -= 4 / 255
            self.controller_changed = True
        if self.code_map[event.code] == ']' and event.value:
            self.steer_offset += 4 / 255
            self.controller_changed = True

        # set speed offset
        if self.code_map[event.code] == '1' and event.value:
            self.speed_offset = 0.15
            self.controller_changed = True
        if self.code_map[event.code] == '2' and event.value:
            self.speed_offset = 0.18
            self.controller_changed = True
        if self.code_map[event.code] == '3' and event.value:
            self.speed_offset = 0.21
            self.controller_changed = True

        # Autonomous
        if self.code_map[event.code] == 'a' and event.value:
            self.is_autonomous = True
            self.controller_changed = True 

        if self.code_map[event.code] == 's' and event.value:
            self.speed = 0
            self.steer = 0
            self.speed_offset = 0
            self.is_autonomous = False
            self.action_changed = True
            self.controller_changed = True

        if self.code_map[event.code] == 'escape' and event.value:
            self.speed = 0
            self.steer = 0
            self.speed_offset = 0
            self.is_autonomous = False
            self.action_changed = True
            self.controller_changed = True
            self.__keep_running = True

    def run(self):
        """Query one set of inputs from the keyboard and send it out."""
        start_time = derp.util.get_timestamp()
        try:
            for event in self.device.read():
                print(event)
                self.process(event)
        except BlockingIOError:
            print('blocking')
            return True
        except Exception as e:
            print(e)
            return True
        if self.controller_changed:
            self.update_controller()
            self.publish(
                "controller",
                isAutonomous=self.is_autonomous,
                speedOffset=self.speed_offset,
                steerOffset=self.steer_offset,
            )
        if self.action_changed:
            self.publish("action", isManual=True, speed=self.speed, steer=self.steer)
            print(self.speed, self.steer)
        return self.__keep_running
