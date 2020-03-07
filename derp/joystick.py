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
from derp.part import Part
import derp.util


class DS4State:
    left_analog_x = 128
    left_analog_y = 128
    right_analog_x = 128
    right_analog_y = 128
    up = 0
    down = 0
    left = 0
    right = 0
    button_square = 0
    button_cross = 0
    button_circle = 0
    button_triangle = 0
    button_l1 = 0
    button_l2 = 0
    button_l3 = 0
    button_r1 = 0
    button_r2 = 0
    button_r3 = 0
    button_share = 0
    button_options = 0
    button_trackpad = 0
    button_ps = 0
    timestamp = 0
    left_trigger = 0
    right_trigger = 0
    accel_y = 0
    accel_x = 0
    accel_z = 0
    orientation_roll = 0
    orientation_yaw = 0
    orientation_pitch = 0
    trackpad_0_id = -1
    trackpad_0_active = False
    trackpad_0_x = 0
    trackpad_0_y = 0
    trackpad_1_id = -1
    trackpad_2_active = False
    trackpad_3_x = 0
    trackpad_4_y = 0
    battery_level = 0
    usb = False
    audio = False
    mic = False

    def __init__(self, recv_buffer=None):
        if recv_buffer:
            self.import_buffer(recv_buffer)

    def import_buffer(self, recv_buffer):
        short = Struct("<h")
        dpad = recv_buffer[7] % 16
        self.left_analog_x = recv_buffer[3]
        self.left_analog_y = recv_buffer[4]
        self.right_analog_x = recv_buffer[5]
        self.right_analog_y = recv_buffer[6]
        self.up = dpad in (0, 1, 7)
        self.down = dpad in (3, 4, 5)
        self.left = dpad in (5, 6, 7)
        self.right = dpad in (1, 2, 3)
        self.button_square = (recv_buffer[7] & 16) != 0
        self.button_cross = (recv_buffer[7] & 32) != 0
        self.button_circle = (recv_buffer[7] & 64) != 0
        self.button_triangle = (recv_buffer[7] & 128) != 0
        self.button_l1 = (recv_buffer[8] & 1) != 0
        self.button_l2 = (recv_buffer[8] & 4) != 0
        self.button_l3 = (recv_buffer[8] & 64) != 0
        self.button_r1 = (recv_buffer[8] & 2) != 0
        self.button_r2 = (recv_buffer[8] & 8) != 0
        self.button_r3 = (recv_buffer[8] & 128) != 0
        self.button_share = (recv_buffer[8] & 16) != 0
        self.button_options = (recv_buffer[8] & 32) != 0
        self.button_trackpad = (recv_buffer[9] & 2) != 0
        self.button_ps = (recv_buffer[9] & 1) != 0
        self.timestamp = recv_buffer[9] >> 2
        self.left_trigger = recv_buffer[10]
        self.right_trigger = recv_buffer[11]
        self.accel_y = short.unpack_from(recv_buffer, 15)[0]
        self.accel_x = short.unpack_from(recv_buffer, 17)[0]
        self.accel_z = short.unpack_from(recv_buffer, 19)[0]
        self.orientation_roll = -(short.unpack_from(recv_buffer, 21)[0])
        self.orientation_yaw = short.unpack_from(recv_buffer, 23)[0]
        self.orientation_pitch = short.unpack_from(recv_buffer, 25)[0]
        self.trackpad_0_id = recv_buffer[37] & 0x7F
        self.trackpad_0_active = (recv_buffer[37] >> 7) == 0
        self.trackpad_0_x = ((recv_buffer[39] & 0x0F) << 8) | recv_buffer[38]
        self.trackpad_0_y = recv_buffer[40] << 4 | ((recv_buffer[39] & 0xF0) >> 4)
        self.trackpad_1_id = recv_buffer[41] & 0x7F
        self.trackpad_2_active = (recv_buffer[41] >> 7) == 0
        self.trackpad_3_x = ((recv_buffer[43] & 0x0F) << 8) | recv_buffer[42]
        self.trackpad_4_y = recv_buffer[44] << 4 | ((recv_buffer[43] & 0xF0) >> 4)
        self.battery_level = recv_buffer[32] % 16
        self.usb = (recv_buffer[32] & 16) != 0
        self.audio = (recv_buffer[32] & 32) != 0
        self.mic = (recv_buffer[32] & 64) != 0


class Joystick(Part):
    """Joystick to drive the car around manually without keyboard."""

    def __init__(self, config):
        """Joystick to drive the car around manually without keyboard."""
        super(Joystick, self).__init__(config, "joystick", [])

        # State/Controls
        self.speed = 0
        self.steer = 0
        self.speed_offset = 0
        self.steer_offset = 0
        self.is_calibrated = True
        self.is_autonomous = False
        self.state = DS4State()
        self.last_state = DS4State()
        self.__fd = None
        self.__input_device = None
        self.__report_fd = None
        self.__report_id = 0x11
        self.__keep_running = True
        self.__connect()

    def __del__(self):
        self.publish("action", isManual=True, speed=0, steer=0)
        self.publish("controller", isAutonomous=False, speedOffset=0, steerOffset=0, exit=True)
        super(Joystick, self).__del__()
        try:
            self.send(red=1, rumble_high=1)
            time.sleep(0.5)
            self.send(blue=0.1, green=0.1, red=0.5)
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
        buf = bytearray(38)
        buf[0] = 0x02
        try:
            return bool(fcntl.ioctl(self.__fd, 3223734279, bytes(buf)))
        except:
            pass
        if self.recv():
            self.update_controller()

    def __in_deadzone(self, value):
        """ Deadzone checker for analog sticks """
        return 128 - self._config["deadzone"] < value <= 128 + self._config["deadzone"]

    def __normalize_stick(self, value, deadzone):
        """
        Normalize stick value from [0, 255] to [0, 1]
        Ignore a 128-centered deadzone
        """
        value -= 128
        value = value - deadzone if value > 0 else value + deadzone
        value /= 127 - deadzone
        return value

    def recv(self, limit=1000, duration=0.001, report_size=78):
        """
        Attempt to get a message from the device. 
        Args:
            limit (int): number of device polls to do
            duration (int): how long to wait between polls
        Returns:
            Whether we have successfully updated the status of the program
        """
        for i in range(limit):
            time.sleep(duration)
            recv_buffer = bytearray(report_size)
            try:
                ret = self.__fd.readinto(recv_buffer)
            except IOError:
                # print("joystick: IO Error")
                continue
            except AttributeError:
                # print("joystick: Attribute Error")
                continue
            if ret is None:
                # print("joystick: ret is none")
                continue
            if ret < report_size:
                # print("joystick: ret too small (%i) expected (%i)" % (ret, report_size))
                continue
            if recv_buffer[0] != self.__report_id:
                # print("joystick: Wrong report id (%i) expected (%i):"
                #      % (recv_buffer[0], self.__report_id))
                continue
            self._timestamp = derp.util.get_timestamp()
            self.last_state = self.state
            self.state = DS4State(recv_buffer)
            self.process_state()
            return True
        return False

    def update_controller(self):
        """Send the state of the system to the controller"""
        green = 1.0 if self.is_autonomous else 0
        red = 1.0 if self.is_calibrated else 0
        blue = 1.0
        light_on = 1.0
        light_off = 0.0
        self.send(red=red, green=green, blue=blue, light_on=light_on, light_off=light_off)
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

    def process_state(self):
        """
        For the given  input, figure out how we should affect the state
        and put that into out.
        """
        self.controller_changed = False
        self.action_changed = False
        self.__keep_running = not self.state.button_trackpad
        if not self.__in_deadzone(self.state.left_analog_x):
            steer = self.__normalize_stick(self.state.left_analog_x, self._config["deadzone"])
            if steer != self.steer:
                self.steer = steer
                self.action_changed = True
        elif not self.__in_deadzone(self.last_state.left_analog_x):
            self.steer = 0
            self.action_changed = True
        if self.state.left_trigger:
            speed = -self.state.left_trigger / 255
            if speed != self.speed:
                self.speed = speed
                self.action_changed = True
        elif self.last_state.left_trigger:
            self.speed = 0
            self.action_changed = True
        if self.state.right_trigger:
            speed = self.state.right_trigger / 255
            if speed != self.speed:
                self.speed = speed
                self.action_changed = True
        elif self.last_state.right_trigger:
            self.speed = 0
            self.action_changed = True
        if self.state.left and not self.last_state.left:
            self.steer_offset -= 1 / 255
            self.controller_changed = True
        if self.state.right and not self.last_state.right:
            self.steer_offset += 1 / 255
            self.controller_changed = True
        if self.state.up and not self.last_state.up:
            self.speed_offset += 5 / 255
            self.controller_changed = True
        if self.state.down and not self.last_state.down:
            self.speed_offset -= 5 / 255
            self.controller_changed = True
        if self.state.button_square and not self.last_state.button_square:
            pass
        if self.state.button_cross and not self.last_state.button_cross:
            self.speed = 0
            self.steer = 0
            self.speed_offset = 0
            self.is_autonomous = False
            self.action_changed = True
            self.controller_changed = True
        if self.state.button_triangle and not self.last_state.button_triangle:
            self.is_autonomous = True
            self.controller_changed = True
        if self.state.button_circle and not self.last_state.button_circle:
            self.controller_changed = True

    def run(self):
        """Query one set of inputs from the joystick and send it out."""
        start_time = derp.util.get_timestamp()
        if not self.recv():
            print("joystick: timed out", start_time)
            self.__connect()
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
        return self.__keep_running
