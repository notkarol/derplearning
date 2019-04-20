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

    def __init__(self, config, state):
        super(Dualshock4, self).__init__(config, state)
        
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
        self.__server_socket.recv_json()
        self.__server_socket.send_json(False)
        sleep(0.1)
        self.__server_socket.disconnect(self.__server_addr)

    def __in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self.__deadzone < value <= 128 + self.__deadzone

    def __normalize_stick(self, value, deadzone):
        """
        Normalize stick value from [0, 255] to [0, 1]
        Ignore a 128-centered deadzone
        """
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value

    def __poll(self):
        """ Request a message from the daemon """
        poller = zmq.Poller()
        poller.register(self.__server_socket, zmq.POLLIN)
        msg = dict(poller.poll(10))
        if len(msg) > 0:
            msgs = self.__server_socket.recv_json()
            self.__last_recv_time = time()
            return msgs
        return []

    def __process(self, status, out):
        """
        For the given status input, figure out how we should affect the state
        and put that into out.
        """

        # linear steering
        if self.__in_deadzone(status['right_analog_x']):
            if self.right_analog_active:
                self.right_analog_active = False
                out['steer'] = 0
        else:
            self.right_analog_active = True
            out['steer'] = self.__normalize_stick(status['right_analog_x'], self.__deadzone)

        # parabolic steering
        if self.__in_deadzone(status['left_analog_x']):
            if self.left_analog_active:
                self.left_analog_active = False
                out['steer'] = 0
        else:
            self.left_analog_active = True
            steer = self.__normalize_stick(status['left_analog_x'], self.__deadzone)
            sign = np.sign(steer)
            steer = abs(steer)
            steer **= self.config['steer_normalizer'][2]
            steer *= self.config['steer_normalizer'][1]
            steer += self.config['steer_normalizer'][0]
            steer = max(0, min(1, steer))
            steer *= sign
            steer = float(steer)
            out['steer'] = steer

        # Forward
        if status['left_trigger']:
            self.left_trigger_active = True
            min_speed, inflect_pt, speed_at_inflect = self.config['speed_elbow']
            speed = status['left_trigger'] / 256
            if speed < inflect_pt:
                speed = (min_speed
                         + speed
                         * (speed_at_inflect - min_speed) 
                         / inflect_pt 
                        )
            else:
                speed = (speed_at_inflect
                         + ( (speed - inflect_pt)
                            * (1 - speed_at_inflect)
                            / (1 - inflect_pt)
                           )
                        )
            out['speed'] = -speed
        elif self.left_trigger_active:
            self.left_trigger_active = False
            out['speed'] = 0

        # Reverse
        if status['right_trigger']:
            self.right_trigger_active = True
            min_speed, inflect_pt, speed_at_inflect = self.config['speed_elbow']
            speed = status['right_trigger'] / 256
            if speed < inflect_pt:
                speed = (min_speed
                         + speed
                         * (speed_at_inflect - min_speed) 
                         / inflect_pt
                        )
            else:
                speed = (speed_at_inflect
                         + (speed - inflect_pt)
                         * (1 - speed_at_inflect)
                         / (1 - inflect_pt)
                        )
            out['speed'] = speed
        elif self.right_trigger_active:
            self.right_trigger_active = False
            out['speed'] = 0
                        
        # Handle buttons
        if status['button_triangle']:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['auto'] = False
            out['use_offset_speed'] = False
        if status['button_ps']:
            out['auto'] = True
            out['use_offset_speed'] = True
        if status['button_cross']:
            out['use_offset_speed'] = True
        if status['button_square']:
            out['record'] = False
        if status['button_circle']:
            if not self.state['record']:
                out['record'] = True

        # Change wheel offset
        if status['left']:
            self.left_active = True
        elif self.left_active:
            self.left_active = False
            out['offset_steer'] = self.state['offset_steer'] - 0.015625
        if status['right']:
            self.right_active = True
        elif self.right_active:
            self.right_active = False
            out['offset_steer'] = self.state['offset_steer'] + 0.015625

        # Fixed speed modifications using arrows
        if status['up']:
            self.up_active = True
        elif self.up_active:
            self.up_active = False
            out['offset_speed'] = self.state['offset_speed'] + 0.015625
        if status['down']:
            self.down_active = True
        elif self.down_active:
            self.down_active = False
            out['offset_speed'] = self.state['offset_speed'] - 0.015625

        # Close down
        if status['button_trackpad'] or status['button_share'] or status['button_options']:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['auto'] = False
            self.state.close()

    def act(self):
        out = {'red': 0,
               'green': 0,
               'blue': 0,
               'light_on': 0,
               'light_off': 0,
               'rumble_high': 0,
               'rumble_low': 0}
               
        # Base color
        if not self.state['record'] and not self.state['auto']:
            out['red'] = 0.130
            out['green'] = 0.085
            out['blue'] = 0.034
            
        # Update based on state
        if self.state['record']:
            out['green'] = 1
        if self.state['use_offset_speed']:
            out['blue'] = 1
        if self.state['auto']:
            out['red'] = 1
            
        self.__server_socket.send_json(out)
        return True

    def sense(self):
        out = {'record' : None,
               'auto' : None,
               'speed' : None,
               'steer' : None,
               'use_offset_speed' : None,
               'offset_speed' : None,
               'offset_steer' : None}

        # Process all outstanding messages
        while True:
            msgs = self.__poll()            
            if len(msgs) == 0:
                # question:If our polling + processing is slower than the stream generation this function might not reach a conclusion, is this possible?
                break
            for msg in msgs:
                self.__process(msg, out)

        # Kill car if we're out of range
        if (time() - self.__last_recv_time) > self.__timeout:
            self.state.close()
            
        # After all messages have been process, update the state
        for field in out:
            if out[field] is not None:
                self.state[field] = out[field]
        return True
