#!/usr/bin/env python3

import numpy as np
import os
import socket
import subprocess
import select
import sys
from time import time, sleep
from struct import Struct
from collections import deque

class DS4Daemon:

    def __init__(self, send_buffer=True, pid_path="/tmp/ds4daemon.pid"):
        """ Initializes the daemon when we connect """

        # Prepare buffers and status variables
        self.__report_id = 0x11
        self.__report_size = 79
        self.__ready = False
        self.__error = False
        self.__send_buffer = send_buffer
        self.__pid_path = pid_path
        self.__deque = deque()
        self.__packet = bytearray(self.__report_size)
        self.__packet[0] = 0x52
        self.__packet[1] = self.__report_id
        self.__packet[2] = 0x80
        self.__packet[4] = 0xFF

        # bluetooth control socket
        self.__ctrl_socket = socket.socket(socket.AF_BLUETOOTH,
                                         socket.SOCK_SEQPACKET,
                                         socket.BTPROTO_L2CAP)
        self.__intr_socket = socket.socket(socket.AF_BLUETOOTH,
                                         socket.SOCK_SEQPACKET,
                                         socket.BTPROTO_L2CAP)

        # Create file to note that we exist
        if self.verifyUnique():
            with open(self.__pid_path, 'w') as f:
                pid = str(os.getpid())
                f.write(pid)


    def verifyUnique(self):
        """ If we're not the only ones running """
        
        # If the PID path doesn't exist then we're unique
        if not os.path.exists(self.__pid_path):
            return True

        # Otherwise check if pid in path exists. If not, delete the file
        with open(self.__pid_path) as f:
            pid = int(f.read())
        try:
            os.kill(pid, 0)
        except OSError:
            os.unlink(self.__pid_path)
            return True

        # Otherwise, it probably doesn't exist so
        return False


    def connect(self):
        """ Try connecting to the controller until we do """
        while True:
            cmd = ["hcitool", "scan", "--flush"]
            res = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf8")
            for _, address, name in [l.split("\t") for l in res.splitlines()[1:]]:
                if name == "Wireless Controller":
                    self.__ctrl_socket.connect((address, 0x11))
                    self.__intr_socket.connect((address, 0x13))
                    self.__intr_socket.setblocking(False)
                    self.__ready = True
                    return


    def __del__(self):
        """ Close all of our sockets and file descriptors """
        self.__ctrl_socket.close()
        self.__intr_socket.close()
        if os.path.exists(self.__pid_path):
            os.unlink(self.__pid_path)


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
        return int(abs(val) * 255) % 256
    
    
    def sendController(self, red=0, green=0, blue=0,
                       light_on=0, light_off=0,
                       rumble_high=0, rumble_low=0):
        self.__packet[7] = self.encodeController(rumble_high)
        self.__packet[8] = self.encodeController(rumble_low)
        self.__packet[9] = self.encodeController(red)
        self.__packet[10] = self.encodeController(green)
        self.__packet[11] = self.encodeController(blue)
        self.__packet[12] = self.encodeController(light_on)
        self.__packet[13] = self.encodeController(light_off)
        self.__ctrl_socket.sendall(self.__packet)
        return True


    def recvController(self):
        ret = -1
        while True:
            buf = bytearray(self.__report_size - 2)
            try:
                ret = self.__intr_socket.recv_into(buf)
                if ret == len(buf) and buf[1] == self.__report_id:
                    self.process(buf, state, out)
            except BlockingIOError as e:
                break

        # Process 'out' into 'state'
        for field in out:
            if out[field] is not None:
                state[field] = out[field]

        return True

def main():
    d = DS4Daemon()
    d.connect()
    print("Connected")

if __name__ == "__main__":
    main()
