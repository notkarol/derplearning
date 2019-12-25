"""
Joystick to drive the car around manually without keyboard.
Inspired by https://github.com/chrippa/ds4drv
"""
import fcntl
from io import FileIO
import os
from struct import Struct
import time
from binascii import crc32
from evdev import InputDevice
from pyudev import Context
import derp.util


class Dualshock4:
    """Joystick to drive the car around manually without keyboard."""
    def __init__(self, config):
        """Joystick to drive the car around manually without keyboard."""
        self.config = config['input']

        # State/Controls
        self.speed = 0
        self.steer = 0
        self.speed_offset = 0
        self.steer_offset = 0
        self.record = False
        self.auto = False

        # Prepare buffers and status variables
        self.status = None
        self.last_status = None
        self.__fd = None
        self.__input_device = None
        self.__report_fd = None
        self.__report_id = 0x11
        self.__report_size = 78
        if not self.__connect():
            return
        while self.__read() is False:
            continue
        self.send(green=1, red=1)
        self.__context, self.__publisher = derp.util.publisher("/tmp/derp_input")

    def __del__(self):
        self.send(blue=1)
        if self.__fd is not None:
            self.__fd.close()
        if self.__report_fd is not None:
            self.__report_fd.close()
        if self.__input_device is not None:
            self.__input_device.ungrab()

    def __find(self):
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
        device_addr, hidraw_device, event_device = self.__find()
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

    def __read(self):
        """Read a message from the controller and process it into a dict"""
        if self.__fd is None:
            return False
        try:
            ret = self.__fd.readinto(self.__buffer)
        except IOError:
            return False
        if ret is None or ret < self.__report_size or self.__buffer[0] != self.__report_id:
            return False

        self.recv_timestamp = derp.util.get_timestamp()
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
            "battery": self.__buffer[32] % 16,
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

    def __process(self):
        """
        For the given status input, figure out how we should affect the state
        and put that into out.
        """
        state_changed = False
        control_changed = False

        # Steer
        if not self.__in_deadzone(self.status["left_analog_x"]):
            steer = self.__normalize_stick(
                self.status["left_analog_x"], self.config["deadzone"]
            )
            if steer != self.steer:
                self.steer = steer
                control_changed = True
        elif not self.__in_deadzone(self.last_status["left_analog_x"]):
            self.steer = 0
            control_changed = True
        if self.status["left_trigger"]:
            speed = -self.status["left_trigger"] / 255
            if speed != self.speed:
                self.speed = speed
                control_changed = True
        elif self.last_status["left_trigger"]:
            self.speed = 0
            control_changed = True
        if self.status["right_trigger"]:
            speed = self.status["right_trigger"] / 255
            if speed != self.speed:
                self.speed = speed
                control_changed = True
        elif self.last_status["right_trigger"]:
            self.speed = 0
            control_changed = True
        if self.status["left"] and not self.last_status["left"]:
            self.steer_offset -= 1 / 255
            state_changed = True
        if self.status["right"] and not self.last_status["right"]:
            self.steer_offset += 1 / 255
            state_changed = True
        if self.status["up"] and not self.last_status["up"]:
            self.speed_offset += 5 / 255
            state_changed = True
        if self.status["down"] and not self.last_status["down"]:
            self.speed_offset -= 5 / 255
            state_changed = True
        if self.status["button_square"] and not self.last_status["button_square"]:
            pass
        if self.status["button_cross"] and not self.last_status["button_cross"]:
            self.speed = 0
            self.steer = 0
            self.speed_offset = 0
            self.record = False
            self.auto = False
            control_changed = True
            state_changed = True
        if self.status["button_triangle"] and not self.last_status["button_triangle"]:
            self.auto = True
            state_changed = True
        if self.status["button_circle"] and not self.last_status["button_circle"]:
            self.record = True
            state_changed = True
        return control_changed, state_changed

    def run(self):
        """Query one set of inputs from the joystick and send it out."""
        while self.__read() is False:
            continue
        control_changed, state_changed = self.__process()
        if state_changed:
            if self.auto:
                self.send(green=1)
            else:
                self.send(green=1, red=1)
            state_message = derp.util.TOPICS['state'].new_message(
                timeCreated=self.recv_timestamp,
                timePublished=derp.util.get_timestamp(),
                speedOffset=self.speed_offset,
                steerOffset=self.steer_offset,
                auto=self.auto,
                record=self.record,
            )
            self.__publisher.send_multipart([b"state", state_message.to_bytes()])
        if control_changed:
            control_messsage = derp.util.TOPICS['control'].new_message(
                timeCreated=self.recv_timestamp,
                timePublished = derp.util.get_timestamp(),
                speed=self.speed,
                steer=self.steer,
                manual=True,
            )
            self.__publisher.send_multipart([b"control", control_message.to_bytes()])
        time.sleep(0.001)

def run(config):
    """Run the joystick in a loop"""
    joystick = Dualshock4(config)
    while True:
        joystick.run()
