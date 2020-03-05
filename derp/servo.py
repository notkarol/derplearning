#!/usr/bin/env python3
import usb.core
import usb.util
from derp.part import Part
import derp.util


class Servo(Part):
    """
    Interface through USB to the servo controller. At the moment the only
    supported capabilities are vague controls of the speed and steering. 
    """

    def __init__(self, config):
        super(Servo, self).__init__(config, "servo", ["brain", "joystick"])
        self.usb_vendor_id = 0x1FFB  # Polulu
        self.usb_product_id = 0x0089  # Maestro 6
        self._device = None
        self._configuration = None
        self.isAutonomous = False
        self.speed_offset = 0
        self.steer_offset = 0
        self.__connect()

    def __del__(self):
        super(Servo, self).__del__()
        if self._device:
            self.reset()

    def reset(self):
        self.__send(self._config["speed_index"], 0, -1, 1)
        self.__send(self._config["steer_index"], 0, -1, 1)

    def __connect(self):
        """ Re-initialize connection to USB servo """
        try:
            self._device = usb.core.find(idVendor=self.usb_vendor_id, idProduct=self.usb_product_id)
            self._configuration = self._device.get_active_configuration()
        except Exception as e:
            return False
        self.reset()
        return True

    def __send(self, value, index, min_value, max_value):
        """ Actually send the message through USB to set the servo to the desired value """
        command = int((1500 + 500 * min(max(value, min_value), max_value)) * 4)
        try:
            self._device.ctrl_transfer(0x40, 0x85, command, index)
        except Exception as e:
            return False
        return True

    def run(self):
        """ Send the servo a specific value in [-1, 1] to move to """
        if self._device is None:
            print("servo: not connected")
            return False
        topic = self.subscribe()
        if topic == "controller":
            self.isAutonomous = self._messages[topic].isAutonomous
            self.speed_offset = self._messages[topic].speedOffset
            self.steer_offset = self._messages[topic].steerOffset
            if self._messages[topic].exit:
                return False
        elif topic == "action":
            if self.isAutonomous or self._messages[topic].isManual:
                self.__send(
                    (self._messages[topic].speed + self.speed_offset)
                    * (-(1 ** self._config["speed_reversed"])),
                    self._config["speed_index"],
                    self._config["speed_min"],
                    self._config["speed_max"],
                )
                self.__send(
                    (self._messages[topic].steer + self.steer_offset)
                    * (-(1 ** self._config["steer_reversed"])),
                    self._config["steer_index"],
                    self._config["steer_min"],
                    self._config["steer_max"],
                )
        return True
