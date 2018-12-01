#!/usr/bin/env python3
import argparse
import os
import zmq
from socket import socket, AF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP
from subprocess import check_output, STDOUT, Popen
from time import sleep, time
from struct import Struct
from collections import deque
import threading
import derp.util

class Daemon:

    def __init__(self, pid_path, port, buffer_max):

        # Prepare buffers and status variables
        self.__port = port
        self.__pid_path = pid_path
        self.__buffer_max = buffer_max
        self.__report_id = 0x11
        self.__report_size = 79
        self.__paired = False
        self.__claimed = False
        self.__device_queue = deque()
        self.__client_queue = deque()
        self.__packet = bytearray(self.__report_size)
        self.__packet[0] = 0x52
        self.__packet[1] = self.__report_id
        self.__packet[2] = 0x80
        self.__packet[4] = 0xFF
        self.__mutex = threading.Lock()
        
        # bluetooth control socket
        self.__ctrl_socket = socket(AF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP)
        self.__intr_socket = socket(AF_BLUETOOTH, SOCK_SEQPACKET, BTPROTO_L2CAP)

        # Create file to note that we exist only if we are the only process
        if not self.verifyUnique():
            print("Other daemon already exists")
            return
        with open(self.__pid_path, 'w') as f:
            pid = str(os.getpid())
            f.write(pid)
        self.__claimed = True

        # Create ZMQ server
        self.__context = zmq.Context()
        self.__client_socket = self.__context.socket(zmq.PAIR)
        self.__client_socket.bind("tcp://*:%s" % self.__port)

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
        if self.__claimed and os.path.exists(self.__pid_path):
            os.unlink(self.__pid_path)

    def paired(self):
        return self.__paired    

    def verifyUnique(self):
        """ If we're not the only ones running """

        # If the PID path doesn't exist then we're unique
        if not os.path.exists(self.__pid_path):
            return True

        # Otherwise check if pid in path exists. If not, delete the file
        with open(self.__pid_path) as f:
            pid = int(f.read())
        print("Checking if pid [%i] exists" % pid)
        try:
            os.kill(pid, 0)
        except OSError:
            print("Deleting [%s]" % self.__pid_path)
            os.unlink(self.__pid_path)
            return True

        # Otherwise, it probably doesn't exist so
        return False

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

    def sendClient(self):
        with self.__mutex:
            out = [self.decodeController(d) for d in self.__client_queue]
        self.__client_socket.send_json(out)
        with self.__mutex:
            self.__client_queue.clear()
        return True

    def recvClient(self):
        self.__last_msg = self.__client_socket.recv_json()
        return self.__last_msg

    def sendController(self, msg=None):
        if type(msg) is not dict:
            msg = {'rumble_high': 0, 'rumble_low': 0,
                   'red': 0.2 if msg is False else 0.05,
                   'green': 0.2 if msg is True else 0.05,
                   'blue': 0.2 if msg is None else 0.05,
                   'light_on': 0, 'light_off': 0}
        self.__packet[7] = self.encodeController(msg['rumble_high'])
        self.__packet[8] = self.encodeController(msg['rumble_low'])
        self.__packet[9] = self.encodeController(msg['red'])
        self.__packet[10] = self.encodeController(msg['green'])
        self.__packet[11] = self.encodeController(msg['blue'])
        self.__packet[12] = self.encodeController(msg['light_on'])
        self.__packet[13] = self.encodeController(msg['light_off'])
        self.__ctrl_socket.sendall(self.__packet)
        return True

    def recvController(self):
        buf = bytearray(self.__report_size - 2)
        with self.__mutex:
            while True:
                try:
                    ret = self.__intr_socket.recv_into(buf)
                    if ret == len(buf) and buf[1] == self.__report_id:
                        self.__client_queue.append(buf)
                        if len(self.__client_queue) > self.__buffer_max:
                            self.__client_queue.popleft()                    
                except BlockingIOError as e:
                    break

            # Check to see what the last message was and what the user typed
            elapsed = time() - self.__creation_time
            if len(self.__client_queue) and not self.__last_msg and elapsed > 5:
                byte8 = self.__client_queue[0][8]
                if byte8 >= 16:
                    self.__creation_time = time()
                    cmd = ["python3", "%s/drive.py" % os.environ['DERP_ROOT'], "--quiet"]
                    if byte8 & 16:
                        cmd.extend(['--controller', 'square'])
                    elif byte8 & 32:
                        pass
                    elif byte8 & 64:
                        cmd.extend(['--controller', 'circle'])
                    elif byte8 & 128:
                        cmd.extend(['--controller', 'triangle'])
                    print(cmd)
                    Popen(cmd)
        return True

    def loop(self):
        threading.Thread(target=self.loopController).start()
        threading.Thread(target=self.loopClient).start()

    def loopController(self):
        while True:
            sleep(0.01)
            self.recvController()

    def loopClient(self):
        while True:
            sleep(0.01)
            self.sendClient()
            msg = self.recvClient()
            self.sendController(msg)


def main(args):
    d = Daemon(pid_path=args.pid_path, port=args.port, buffer_max=args.buffer_max)
    if d.paired():
        d.loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid_path', type=str, default="/tmp/ds4daemon.pid",
                        help="location of pid path to verify that other daemons are not running")
    parser.add_argument('--port', type=int, default=2455,
                        help="port number that the client will be connecting to")
    parser.add_argument('--buffer_max', type=int, default=10,
                        help="max number of messages from the controller to store at any one time")
    args = parser.parse_args()
    main(args)
