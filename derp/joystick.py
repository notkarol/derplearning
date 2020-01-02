"""
Joystick to drive the car around manually without keyboard.
Inspired by https://github.com/chrippa/ds4drv
"""
import fcntl
from io import FileIO
import os
from struct import Struct
import time
from evdev import InputDevice
from binascii import crc32
from pyudev import Context
import derp.util


class Dualshock4:
    """Joystick to drive the car around manually without keyboard."""
    def __init__(self, config):
        """Joystick to drive the car around manually without keyboard."""
        self.config = config['joystick']

        # State/Controls
        self.speed = 0
        self.steer = 0
        self.speed_offset = 0
        self.steer_offset = 0
        self.calibrated = False
        self.is_recording = False
        self.is_autonomous = False
        self.recv_timestamp = derp.util.get_timestamp()

        # Prepare buffers and status variables
        self.status = {
            "left_analog_x": 128,
            "left_analog_y": 128,
            "right_analog_x": 128,
            "right_analog_y": 128,
            "up": 0,
            "down": 0,
            "left": 0,
            "right": 0,
            "button_square": 0,
            "button_cross": 0, 
            "button_circle": 0,
            "button_triangle": 0,
            "button_l1": 0,
            "button_l2": 0,
            "button_l3": 0,
            "button_r1": 0,
            "button_r2": 0,
            "button_r3": 0,
            "button_share": 0,
            "button_options": 0,
            "button_trackpad": 0,
            "button_ps": 0,
            "timestamp": 0,
            "left_trigger": 0,
            "right_trigger": 0,
            "accel_y": 0,
            "accel_x": 0,
            "accel_z": 0,
            "orientation_roll": 0,
            "orientation_yaw": 0,
            "orientation_pitch": 0,
            "trackpad_0_id": -1,
            "trackpad_0_active": False,
            "trackpad_0_x": 0,
            "trackpad_0_y": 0,
            "trackpad_1_id": -1,
            "trackpad_2_active": False,
            "trackpad_3_x": 0,
            "trackpad_4_y": 0,
            "battery_level": 0,
            "usb": False,
            "audio": False,
            "mic": False,
        }
        self.last_status = self.status
        self.__fd = None
        self.__input_device = None
        self.__report_fd = None
        self.__report_id = 0x11
        self.__report_size = 78
        self.is_connected = self.__connect()
        self.keep_running = self.is_connected
        self.__context, self.__publisher = derp.util.publisher("/tmp/derp_joystick")

    def __del__(self):
        try:
            self.send(red=1, rumble_high=1)
            time.sleep(0.5)
            self.send(blue=1)
        except:
            pass
        if self.__fd is not None:
            self.__fd.close()
        if self.__input_device is not None:
            self.__input_device.ungrab()

    def __find_device(self):
        context = Context()
        for hidraw_device in context.list_devices(subsystem="hidraw"):
            hid_device = hidraw_device.parent
            if hid_device.subsystem != "hid" or hid_device.get("HID_NAME") != "Wireless Controller":
                continue
            for child in hid_device.parent.children:
                event_device = child.get("DEVNAME", "")
                if event_device.startswith("/dev/input/event"):
                    break
            else:
                continue

            device_addr = hid_device.get("HID_UNIQ", "").upper()
            return device_addr, hidraw_device.device_node, event_device
        return None, None, None

    def __connect(self):
        device_addr, hidraw_device, event_device = self.__find_device()
        if device_addr is None:
            return False
        self.__report_fd = os.open(hidraw_device, os.O_RDWR | os.O_NONBLOCK)
        self.__fd = FileIO(self.__report_fd, "rb+", closefd=False)
        self.__input_device = InputDevice(event_device)
        self.__input_device.grab()
        self.__buffer = bytearray(self.__report_size)
        buf = bytearray(38)
        buf[0] = 0x02
        try:
            return bool(fcntl.ioctl(self.__fd, 3223734279, bytes(buf)))
        except:
            pass
        return False

    def __in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self.config["deadzone"] < value <= 128 + self.config["deadzone"]

    def __normalize_stick(self, value, deadzone):
        """
        Normalize stick value from [0, 255] to [0, 1]
        Ignore a 128-centered deadzone
        """
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value

    def poll_device(self):
        """Read a message from the controller and process it into a dict"""
        try:
            ret = self.__fd.readinto(self.__buffer)
        except IOError:
            return False
        except AttributeError:
            return False
        if ret is None or ret < self.__report_size or self.__buffer[0] != self.__report_id:
            return False
        return True

    def recv(self, limit=100, duration=0.001):
        """
        Attempt to get a message from the device. 
        Args:
            limit (int): number of device polls to do
            duration (int): how long to wait between polls
        Returns:
            Whether we have successfully updated the status of the program
        """
        for i in range(limit):
            if self.poll_device():
                self.recv_timestamp = derp.util.get_timestamp()
                self.update_status()
                return True
            time.sleep(duration)
        return False

    def update_status(self):
        self.last_status = self.status
        short = Struct("<h")
        dpad = self.__buffer[7] % 16
        self.status = {
            "left_analog_x": self.__buffer[3],
            "left_analog_y": self.__buffer[4],
            "right_analog_x": self.__buffer[5],
            "right_analog_y": self.__buffer[6],
            "up": (dpad in (0, 1, 7)),
            "down": (dpad in (3, 4, 5)),
            "left": (dpad in (5, 6, 7)),
            "right": (dpad in (1, 2, 3)),
            "button_square": (self.__buffer[7] & 16) != 0,
            "button_cross": (self.__buffer[7] & 32) != 0,
            "button_circle": (self.__buffer[7] & 64) != 0,
            "button_triangle": (self.__buffer[7] & 128) != 0,
            "button_l1": (self.__buffer[8] & 1) != 0,
            "button_l2": (self.__buffer[8] & 4) != 0,
            "button_l3": (self.__buffer[8] & 64) != 0,
            "button_r1": (self.__buffer[8] & 2) != 0,
            "button_r2": (self.__buffer[8] & 8) != 0,
            "button_r3": (self.__buffer[8] & 128) != 0,
            "button_share": (self.__buffer[8] & 16) != 0,
            "button_options": (self.__buffer[8] & 32) != 0,
            "button_trackpad": (self.__buffer[9] & 2) != 0,
            "button_ps": (self.__buffer[9] & 1) != 0,
            "timestamp": self.__buffer[9] >> 2,
            "left_trigger": self.__buffer[10],
            "right_trigger": self.__buffer[11],
            "accel_y": short.unpack_from(self.__buffer, 15)[0],
            "accel_x": short.unpack_from(self.__buffer, 17)[0],
            "accel_z": short.unpack_from(self.__buffer, 19)[0],
            "orientation_roll": -(short.unpack_from(self.__buffer, 21)[0]),
            "orientation_yaw": short.unpack_from(self.__buffer, 23)[0],
            "orientation_pitch": short.unpack_from(self.__buffer, 25)[0],
            "trackpad_0_id": self.__buffer[37] & 0x7F,
            "trackpad_0_active": (self.__buffer[37] >> 7) == 0,
            "trackpad_0_x": ((self.__buffer[39] & 0x0F) << 8) | self.__buffer[38],
            "trackpad_0_y": self.__buffer[40] << 4 | ((self.__buffer[39] & 0xF0) >> 4),
            "trackpad_1_id": self.__buffer[41] & 0x7F,
            "trackpad_2_active": (self.__buffer[41] >> 7) == 0,
            "trackpad_3_x": ((self.__buffer[43] & 0x0F) << 8) | self.__buffer[42],
            "trackpad_4_y": self.__buffer[44] << 4 | ((self.__buffer[43] & 0xF0) >> 4),
            "battery_level": self.__buffer[32] % 16,
            "usb": (self.__buffer[32] & 16) != 0,
            "audio": (self.__buffer[32] & 32) != 0,
            "mic": (self.__buffer[32] & 64) != 0,
        }
        return True

    def update_controller(self):
        """Send the state of the system to the controller"""
        red = 1.0 if self.is_recording else 0.5
        green = 1.0 if self.is_autonomous else 0.5
        light_on = 0.0 if self.calibrated else 0.8
        light_off = 0.0 if self.calibrated else 0.8
        self.send(red=red, green=green, blue=0.5, light_on=light_on, light_off=light_off)

    def publish_controller(self):
        message = derp.util.TOPICS['controller'].new_message(
            timeCreated=self.recv_timestamp,
            timePublished=derp.util.get_timestamp(),
            isRecording=self.is_recording,
            isAutonomous=self.is_autonomous,
            speedOffset=self.speed_offset,
            steerOffset=self.steer_offset,
        )
        self.__publisher.send_multipart([b"controller", message.to_bytes()])

    def publish_action(self):
        message = derp.util.TOPICS['action'].new_message(
            timeCreated=self.recv_timestamp,
            timePublished = derp.util.get_timestamp(),
            isManual=True,
            speed=self.speed,
            steer=self.steer,
        )
        self.__publisher.send_multipart([b"action", message.to_bytes()])

    def send(self, rumble_high=0, rumble_low=0, red=0, green=0, blue=0, light_on=0, light_off=0):
        """Actuate the controller by setting its rumble or light color/blink"""
        packet = bytearray(79)
        packet[:5] = [0xA2, 0x11, 0x80, 0x00, 0xFF]
        packet[7] = int(rumble_high * 255 + 0.5)
        packet[8] = int(rumble_low * 255 + 0.5)
        packet[9] = int(red * 255 + 0.5)
        packet[10] = int(green * 255 + 0.5)
        packet[11] = int(blue * 255 + 0.5)
        packet[12] = int(light_on * 255 + 0.5)
        packet[13] = int(light_off * 255 + 0.5)
        crc = crc32(packet[:-4])
        packet[-4] = crc & 0x000000FF
        packet[-3] = (crc & 0x0000FF00) >> 8
        packet[-2] = (crc & 0x00FF0000) >> 16
        packet[-1] = (crc & 0xFF000000) >> 24
        hid = bytearray((self.__report_id,))
        if self.__fd is not None:
            self.__fd.write(hid + packet[2:])
            return True
        return False

    def process_status(self):
        """
        For the given status input, figure out how we should affect the state
        and put that into out.
        """
        self.controller_changed = False
        self.action_changed = False
        self.keep_running = not self.status['button_trackpad']
        if not self.__in_deadzone(self.status["left_analog_x"]):
            steer = self.__normalize_stick(
                self.status["left_analog_x"], self.config["deadzone"]
            )
            if steer != self.steer:
                self.steer = steer
                self.action_changed = True
        elif not self.__in_deadzone(self.last_status["left_analog_x"]):
            self.steer = 0
            self.action_changed = True
        if self.status["left_trigger"]:
            speed = -self.status["left_trigger"] / 255
            if speed != self.speed:
                self.speed = speed
                self.action_changed = True
        elif self.last_status["left_trigger"]:
            self.speed = 0
            self.action_changed = True
        if self.status["right_trigger"]:
            speed = self.status["right_trigger"] / 255
            if speed != self.speed:
                self.speed = speed
                self.action_changed = True
        elif self.last_status["right_trigger"]:
            self.speed = 0
            self.action_changed = True
        if self.status["left"] and not self.last_status["left"]:
            self.steer_offset -= 1 / 255
            self.controller_changed = True
        if self.status["right"] and not self.last_status["right"]:
            self.steer_offset += 1 / 255
            self.controller_changed = True
        if self.status["up"] and not self.last_status["up"]:
            self.speed_offset += 5 / 255
            self.controller_changed = True
        if self.status["down"] and not self.last_status["down"]:
            self.speed_offset -= 5 / 255
            self.controller_changed = True
        if self.status["button_square"] and not self.last_status["button_square"]:
            pass
        if self.status["button_cross"] and not self.last_status["button_cross"]:
            self.speed = 0
            self.steer = 0
            self.speed_offset = 0
            self.is_recording = False
            self.is_autonomous = False
            self.action_changed = True
            self.controller_changed = True
        if self.status["button_triangle"] and not self.last_status["button_triangle"]:
            self.is_autonomous = True
            self.controller_changed = True
        if self.status["button_circle"] and not self.last_status["button_circle"]:
            self.is_recording = True
            self.controller_changed = True            

    def run(self):
        """Query one set of inputs from the joystick and send it out."""
        start_time = derp.util.get_timestamp()
        if not self.recv():
            print("joystick: timed out")
            return False
        self.process_status()
        if self.controller_changed:
            self.update_controller()
            self.publish_controller()
        if self.action_changed:
            self.publish_action()
        return self.keep_running

def run(config):
    """Run the joystick in a loop"""
    joystick = Dualshock4(config)
    while joystick.run():
        pass
