#!/usr/bin/env python3

import numpy as np
import os
from socket import socket, AF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP
from subprocess import check_output, STDOUT, Popen
import socket
import select
import sys
import zmq
from struct import Struct
from collections import deque
import threading
from time import time, sleep
import derp.util


class Dualshock4:

    def __init__(self, config):
        self.config = config
        
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

        # Prepare buffers and status variables
        self.__buffer_max = buffer_max
        self.__report_id = 0x11
        self.__report_size = 79
        self.__paired = False
        self.__claimed = False
        self.__client_queue = deque()
        self.__packet = bytearray(self.__report_size)
        self.__packet[0] = 0x52
        self.__packet[1] = self.__report_id
        self.__packet[2] = 0x80
        self.__packet[4] = 0xFF
        
        # bluetooth control socket
        self.__ctrl_socket = socket(AF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP)
        self.__intr_socket = socket(AF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP)

        # Pair and send messages
        if not self.pair():
            print("Count not pair")
            return

        self.__last_msg = None
        self.__creation_time = 0
        self.sendController(None)

    def __del__(self):
        """ Close all of our sockets and file descriptors """
        self.__ctrl_socket.close()
        self.__intr_socket.close()

    def is_paired(self):
        return self.__paired    


    def pair(self, max_attempts=5):
        """ Try pairing with the controller """
        if not self.__claimed:
            return False

        # Try conn
        for attempt in range(max_attempts):
            print("Attempt [%i] of [%i] to pair controller" % (attempt + 1, max_attempts))
            cmd = ["hcitool", "scan", "--flush"]
            res = check_output(cmd, stderr=STDOUT).decode("utf8")
            for _, address, name in [l.split("\t") for l in res.splitlines()[1:]]:
                if name == "Wireless Controller":
                    print("Found controller at [%s]" % address)
                    self.__ctrl_socket.connect((address, 0x11))
                    self.__intr_socket.connect((address, 0x13))
                    self.__intr_socket.setblocking(False)
                    self.__paired = True
                    # Does not receive packets properly until we send it at least one command
                    return True
        return False

    def decodeController(self, buf):
        """
        Decodes a buffer from the controller into a dict
        """
        short = Struct("<h")
        dpad = buf[8] % 16
        status = {"left_analog_x" : buf[4],
                  "left_analog_y" : buf[5],
                  "right_analog_x" : buf[6],
                  "right_analog_y" : buf[7],
                  "up" :  (dpad in (0, 1, 7)),
                  "down" : (dpad in (3, 4, 5)),
                  "left" : (dpad in (5, 6, 7)),
                  "right" : (dpad in (1, 2, 3)),
                  "button_square" : (buf[8] & 16) != 0,
                  "button_cross" : (buf[8] & 32) != 0,
                  "button_circle" : (buf[8] & 64) != 0,
                  "button_triangle" : (buf[8] & 128) != 0,
                  "button_l1" : (buf[9] & 1) != 0,
                  "button_l2" : (buf[9] & 4) != 0,
                  "button_l3" : (buf[9] & 64) != 0,
                  "button_r1" : (buf[9] & 2) != 0,
                  "button_r2" : (buf[9] & 8) != 0,
                  "button_r3" : (buf[9] & 128) != 0,
                  "button_share" : (buf[9] & 16) != 0,
                  "button_options" : (buf[9] & 32) != 0,
                  "button_trackpad" :  (buf[10] & 2) != 0,
                  "button_ps" : (buf[10] & 1) != 0,
                  "timestamp" : buf[10] >> 2,
                  "left_trigger" : buf[11],
                  "right_trigger" : buf[12],
                  "battery" : buf[15] % 16,
                  "accel_y" : short.unpack_from(buf, 16)[0], 
                  "accel_x" : short.unpack_from(buf, 18)[0], 
                  "accel_z" : short.unpack_from(buf, 20)[0], 
                  "roll" : -(short.unpack_from(buf, 22)[0]),
                  "yaw" : short.unpack_from(buf, 24)[0],
                  "pitch" : short.unpack_from(buf, 26)[0],
                  "battery_level" : buf[33] % 16,
                  "usb" : (buf[33] & 16) != 0,
                  "audio" : (buf[33] & 32) != 0,
                  "mic" : (buf[33] & 64) != 0}
        return status

    def encodeController(self, val):
        val = float(val) * 255 # normalize from 0->1 to 0->255
        val = int(val + 0.5) # round it to nearest int
        if val < 0: val = 0 # bound it
        if val > 255: val = 255 # bound it
        return val

    def sendController(self, rumble_high=0, rumble_low=0, red=0, green=0, blue=0,
                       light_on_dur=0, light_off_dur=0):
        self.__packet[7] = self.encodeController(rumble_high)
        self.__packet[8] = self.encodeController(rumble_low)
        self.__packet[9] = self.encodeController(red)
        self.__packet[10] = self.encodeController(green)
        self.__packet[11] = self.encodeController(blue)
        self.__packet[12] = self.encodeController(light_on_dur)
        self.__packet[13] = self.encodeController(light_off_dur)
        return self.__ctrl_socket.sendall(self.__packet)

    def recvController(self):
        buf = bytearray(self.__report_size - 2)
        try:
            ret = self.__intr_socket.recv_into(buf)
            if ret == len(buf) and buf[1] == self.__report_id:
                return None
        except BlockingIOError as e:
            return
        return self.decodeController(buf)

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
            out['offset_steer'] = self.state['offset_steer'] - 4 / 256
        if status['right']:
            self.right_active = True
        elif self.right_active:
            self.right_active = False
            out['offset_steer'] = self.state['offset_steer'] + 4 / 256

        # Fixed speed modifications using arrows
        if status['up']:
            self.up_active = True
        elif self.up_active:
            self.up_active = False
            out['offset_speed'] = self.state['offset_speed'] + 4 / 256
        if status['down']:
            self.down_active = True
        elif self.down_active:
            self.down_active = False
            out['offset_speed'] = self.state['offset_speed'] - 4 / 256

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
