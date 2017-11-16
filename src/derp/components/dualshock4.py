#!/usr/bin/env python3

import numpy as np
import os
import socket
import subprocess
import select
import sys
from time import time, sleep
from derp.component import Component
from struct import Struct
import derp.util as util

class Dualshock4(Component):

    def __init__(self, config):
        super(Dualshock4, self).__init__(config)

        # bluetooth control socket
        self.report_id = 0x11
        self.report_size = 79
        self.ctrl_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET,
                                         socket.BTPROTO_L2CAP)
        self.intr_socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET,
                                         socket.BTPROTO_L2CAP)

        # Prepare packet
        self.packet = bytearray(79)
        self.packet[0] = 0x52
        self.packet[1] = self.report_id
        self.packet[2] = 0x80
        self.packet[4] = 0xFF

        # The deadzone of the analog sticks to ignore them
        self.deadzone = self.config['deadzone']        
        self.left_analog_active = False
        self.right_analog_active = False
        self.left_trigger_active = False
        self.right_trigger_active = False
        self.up_active = False
        self.down_active = False
        self.left_active = False
        self.right_active = False


    def __del__(self):
        """ Close all of our sockets and file descriptors """
        self.ctrl_socket.close()
        self.intr_socket.close()


    def in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self.deadzone < value <= 128 + self.deadzone


    def act(self, state):

        # Prepare command
        rumble = [0, 0]
        flash = [0, 0]
        
        # Base color
        if not state['record'] and not state['auto_speed'] and not state['auto_steer']:
            rgb = [0.1, 0.1, 0.1]
        else:
            rgb = [0, 0, 0]
            
        # Update based on state
        if state['record']:
            flash[0] = 0.3
            flash[1] = 0.1
            rgb[1] = 1
        if state['auto_steer']:
            rgb[0] = 0.5
        if state['auto_speed']:
            rgb[2] = 0.5
            
        # Prepare and sendpacket
        self.packet[7] = int(rumble[0] * 255)
        self.packet[8] = int(rumble[1] * 255)
        self.packet[9] = int(rgb[0] * 255)
        self.packet[10] = int(rgb[1] * 255)
        self.packet[11] = int(rgb[2] * 255)
        self.packet[12] = int(flash[0] * 255)
        self.packet[13] = int(flash[1] * 255)
        self.ctrl_socket.sendall(self.packet)
        return True
    

    def discover(self):
        """
        Find and initialize the available devices
        """

        # Make sure we can send commands
        n_attemps = 5
        for attempt in range(n_attemps):
            print("Attempt %i of %i" % (attempt + 1, n_attemps), end='\r')
            cmd = ["hcitool", "scan", "--flush"]
            res = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf8")
            for _, address, name in [l.split("\t") for l in res.splitlines()[1:]]:
                if name == "Wireless Controller":
                    self.ctrl_socket.connect((address, 0x11))
                    self.intr_socket.connect((address, 0x13))
                    self.intr_socket.setblocking(False)
                    return True
        
        return False

    
    def scribe(self, state):        
        return True

    
    def __normalize_stick(self, value, deadzone):
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value
    

    # Prepare status based on buffer
    def __prepare(self, buf):
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
            
    
    def __process(self, buf, state, out):
        status = self.__prepare(buf)

        # Left Analog Steering
        if self.in_deadzone(status['left_analog_x']):
            if self.left_analog_active:
                self.left_analog_active = False
                out['steer'] = 0
        else:
            self.left_analog_active = True
            out['steer'] = self.__normalize_stick(status['left_analog_x'], self.deadzone)

        # Right Analog Steering
        if self.in_deadzone(status['right_analog_x']):
            if self.right_analog_active:
                self.right_analog_active = False
                out['steer'] = 0
        else:
            self.right_analog_active = True
            steer = self.__normalize_stick(status['right_analog_x'], self.deadzone)
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
            out['folder'] = False
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
            out['folder'] = False
        if status['button_circle']:
            if not state['record']:
                out['record'] = True
                out['folder'] = util.get_record_folder()

        # Change wheel offset
        if status['left']:
            self.left_active = True
        elif self.left_active:
            self.left_active = False
            out['steer_offset'] = state['steer_offset'] - 1 / 128
        if status['right']:
            self.right_active = True
        elif self.right_active:
            self.right_active = False
            out['steer_offset'] = state['steer_offset'] + 1 / 128

        # Fixed speed modifications using arrows
        if status['up']:
            self.up_active = True
        elif self.up_active:
            self.up_active = False
            out['speed'] = state['speed'] + 0.015625
        if status['down']:
            self.down_active = True
        elif self.down_active:
            out['speed'] = state['speed'] - 0.015625

        # Close down
        if status['button_trackpad']:
            out['speed'] = 0
            out['steer'] = 0
            out['record'] = False
            out['folder'] = False
            out['auto_speed'] = False
            out['auto_steer'] = False
            out['exit'] = True

           
    def sense(self, state):
        ret = -1
        buf = bytearray(77)
        out = { 'speed' : None,
                'steer' : None,
                'record' : None,
                'folder' : None,
                'auto_speed' : None,
                'auto_steer' : None,
                'steer_offset' : None,
                'exit' : None }

        # Fetch input messages and process them. Store it in out
        while True:
            try:
                ret = self.intr_socket.recv_into(buf)
                if ret == len(buf) and buf[1] == self.report_id:
                    self.__process(buf, state, out)
            except BlockingIOError as e:
                break

        # Process 'out' into 'state'
        for field in out:
            if out[field] is not None:
                state[field] = out[field]
        return True
            
    
    def write(self):        
        return True    
