#!/usr/bin/env python3

import numpy as np
import os
import socket
import subprocess
import select
import sys
import zmq
from time import time, sleep
from derp.component import Component
import derp.util as util

class Dualshock4(Component):

    def __init__(self, config, full_config):
        super(Dualshock4, self).__init__(config, full_config)
        
        # The deadzone of the analog sticks to ignore them
        self.__timeout = self.config['timeout']
        self.__deadzone = self.config['deadzone']
        self.left_analog_active = False
        self.right_analog_active = False
        self.left_trigger_active = False
        self.right_trigger_active = False
        self.up_active = False
        self.down_active = False
        self.left_active = False
        self.right_active = False

        self.__port = 2455
        self.__server_addr = "tcp://localhost:%s" % self.__port
        self.__context = zmq.Context()
        self.__server_socket = self.__context.socket(zmq.PAIR)
        self.__server_socket.connect(self.__server_addr)
        self.ready = True
        self.__last_recv_time = 0

        # Reset the message queue
        self.__server_socket.send_json(True)

        
    def __del__(self):
        """ Close all of our sockets and file descriptors """
        print("SENDING MESSAGE TO SLEEP")
        self.__server_socket.recv_json()
        self.__server_socket.send_json(False)
        sleep(0.1)
        self.__server_socket.disconnect(self.__server_addr)
        super(Dualshock4, self).__del__()


    def in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self.__deadzone < value <= 128 + self.__deadzone


    def normalize_stick(self, value, deadzone):
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value


    def process(self, status, state, out):

        # Left Analog Steering
        if self.in_deadzone(status['left_analog_x']):
            if self.left_analog_active:
                self.left_analog_active = False
                out['steer'] = 0
        else:
            self.left_analog_active = True
            out['steer'] = self.normalize_stick(status['left_analog_x'], self.__deadzone)

        # Right Analog Steering
        if self.in_deadzone(status['right_analog_x']):
            if self.right_analog_active:
                self.right_analog_active = False
                out['steer'] = 0
        else:
            self.right_analog_active = True
            steer = self.normalize_stick(status['right_analog_x'], self.__deadzone)
            sign = np.sign(steer)
            steer = abs(steer)
            steer *= self.config['steer_normalizer'][1]
            steer **= self.config['steer_normalizer'][2]
            steer += self.config['steer_normalizer'][0]
            steer = max(0, min(1, steer))
            steer *= sign
            steer = float(steer)
            out['steer'] = steer

        # Speed
        if status['left_trigger']:
            self.left_trigger_active = True
            z, x, y = self.config['speed_elbow']
            speed = status['left_trigger'] / 256
            if speed < x:
                speed = z + speed * (y - z) / x
            else:
                speed = (y + (speed - x) * (1 - y) / (1 - x))
            out['speed'] = speed
        elif self.left_trigger_active:
            self.left_trigger_active = False
            out['speed'] = 0

        # Handle speed reverse
        if status['right_trigger']:
            self.right_trigger_active = True
            z, x, y = self.config['speed_elbow']
            speed = status['right_trigger'] / 256
            if speed < x:
                speed = z + speed * (y - z) / x
            else:
                speed = (y + (speed - x) * (1 - y) / (1 - x))
            out['speed'] = -speed
        elif self.right_trigger_active:
            self.right_trigger_active = False
            out['speed'] = 0
                        
        # Handle buttons
        if status['button_triangle']:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['auto_speed'] = False
            out['auto_steer'] = False
        if status['button_share']:
            out['auto_speed'] = True
        if status['button_options']:
            out['auto_steer'] = True
        if status['button_ps']:
            out['auto_speed'] = True
            out['auto_steer'] = True
        if status['button_cross']:
            out['auto_speed'] = False
            out['auto_steer'] = False
        if status['button_square']:
            out['record'] = False
        if status['button_circle']:
            if not state['record']:
                out['record'] = True

        # Change wheel offset
        if status['left']:
            self.left_active = True
        elif self.left_active:
            self.left_active = False
            out['offset_steer'] = state['offset_steer'] - 1 / 128
        if status['right']:
            self.right_active = True
        elif self.right_active:
            self.right_active = False
            out['offset_steer'] = state['offset_steer'] + 1 / 128

        # Fixed speed modifications using arrows
        if status['up']:
            self.up_active = True
        elif self.up_active:
            self.up_active = False
            out['offset_speed'] = state['offset_speed'] + 0.1
        if status['down']:
            self.down_active = True
        elif self.down_active:
            out['offset_speed'] = state['offset_speed'] - 0.1

        # Close down
        if status['button_trackpad']:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['auto_speed'] = False
            out['auto_steer'] = False
            state.close()


    def act(self, state):
        out = {'red': 0,
               'green': 0,
               'blue': 0,
               'light_on': 0,
               'light_off': 0,
               'rumble_high': 0,
               'rumble_low': 0}
               
        # Base color
        if (('record' in state and not state['record']) and
            ('auto_speed' in state and not state['auto_speed']) and
            ('auto_steer' in state and not state['auto_steer'])):
            out['red'] = 0.130
            out['green'] = 0.085
            out['blue'] = 0.034
            
        # Update based on state
        if 'record' in state and state['record']:
            out['green'] = 1
        if 'auto_steer' in state and state['auto_steer']:
            out['red'] = 1
        if 'auto_steer' in state and state['auto_speed']:
            out['blue'] = 1

        self.__server_socket.send_json(out)
        return True


    def sense(self, state):
        out = {'record' : None,
               'auto_speed' : None,
               'auto_steer' : None,
               'speed' : None,
               'steer' : None,
               'offset_speed' : None,
               'offset_steer' : None}

        # Ask the controller for messages
        poller = zmq.Poller()
        poller.register(self.__server_socket, zmq.POLLIN)
        msg = dict(poller.poll(10))
        if len(msg) > 0:
            msgs = self.__server_socket.recv_json()
            self.__last_recv_time = time()
        else:
            msgs = []

        # For each message, process the fields we want to set a new desired value
        for msg in msgs:
            self.process(msg, state, out)

        # Kill car if we're out of range
        if (time() - self.__last_recv_time) > self.__timeout:
            state.close()
            
        # After all messages have been processed, update the state
        for field in out:
            if out[field] is not None:
                state[field] = out[field]
        return True
