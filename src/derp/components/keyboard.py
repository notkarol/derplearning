#!/usr/bin/env python3

import evdev
import os
from time import time
from derp.component import Component
import derp.util


def find_device(names):
    for filename in sorted(evdev.list_devices()):
        device = evdev.InputDevice(filename)
        device_name = device.name.lower()
        for name in names:
            if name in device_name:
                return device
    return None  

class Keyboard(Component):

    def __init__(self, config, full_config):
        super(Keyboard, self).__init__(config, full_config)
        self.device = None

        # Prepare key code
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

        self.device = find_device(['keyboard', 'kbd'])
        self.ready = self.device is not None

    def __del__(self):
        super(Keyboard, self).__del__()
        if self.device is not None:
            self.device.close()
            self.device = None


    def process(self, state, out, event):

        # Skip events that I don't know what they mean, but appear all the time
        if event.code == 0 or event.code == 4:
            return

        # Set steer
        if self.code_map[event.code] == 'arrow_left' and event.value:
            out['steer'] = state['steer'] - 0.1
            return
        if self.code_map[event.code] == 'arrow_right' and event.value:
            out['steer'] = state['steer'] + 0.1
            return

        # Set speed
        if self.code_map[event.code] == 'arrow_up' and event.value:
            out['speed'] = state['speed'] + 0.01
            return
        if self.code_map[event.code] == 'arrow_down' and event.value:
            out['speed'] = state['speed'] - 0.01
            return
        
        # set steer offset
        if self.code_map[event.code] == '[' and event.value:
            out['offset_steer'] = state['offset_steer'] - 0.00390625
            return
        if self.code_map[event.code] == ']' and event.value:
            out['offset_steer'] = state['offset_steer'] + 0.00390625
            return

        # set speed offset
        if self.code_map[event.code] == '1' and event.value:
            out['offset_speed'] = 0.10
            return
        if self.code_map[event.code] == '2' and event.value:
            out['offset_speed'] = 0.12
            return
        if self.code_map[event.code] == '3' and event.value:
            out['offset_speed'] = 0.14
            return
        if self.code_map[event.code] == '4' and event.value:
            out['offset_speed'] = 0.16
            return
        if self.code_map[event.code] == '5' and event.value:
            out['offset_speed'] = 0.18
            return
        if self.code_map[event.code] == '6' and event.value:
            out['offset_speed'] = 0.20
            return
        if self.code_map[event.code] == '7' and event.value:
            out['offset_speed'] = 0.22
            return
        if self.code_map[event.code] == '8' and event.value:
            out['offset_speed'] = 0.24
            return
        if self.code_map[event.code] == '9' and event.value:
            out['offset_speed'] = 0.26
            return
        if self.code_map[event.code] == '0' and event.value:
            out['offset_speed'] = 0.28
            return

        # Record
        if self.code_map[event.code] == 'r' and event.value:
            out['record'] = True
            return

        # Autonomous
        if self.code_map[event.code] == 'a' and event.value:
            out['auto'] = True
            return
        if self.code_map[event.code] == 's' and event.value:
            out['use_offset_speed'] = True
            return


        # Stop car and recording, but keep running program
        if self.code_map[event.code] == 'q' and event.value:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['auto'] = False
            out['use_offset_speed'] = False

        # Exit
        if self.code_map[event.code] == 'escape' and event.value:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['auto'] = False
            out['use_offset_speed'] = False
            state.close()
            return


    def sense(self, state):
        out = {'record' : None,
               'speed' : None,
               'steer' : None,
               'auto' : None,
               'use_offset_speed' : None,
               'offset_speed' : None,
               'offset_steer' : None}

        # Process every action we received until there are no more left
        try:
            for event in self.device.read():
                self.process(state, out, event)
        except BlockingIOError:
            pass

        # Process 'out' into 'state'
        for field in out:
            if out[field] is not None:
                state[field] = out[field]
        return True
