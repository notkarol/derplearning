#!/usr/bin/env python3
import usb.core
import usb.util
import derp.util

class Servo:
    """
    Interface through USB to the servo controller. At the moment the only
    supported capabilities are vague controls of the speed and steering. 
    """

    def __init__(self, config):
        self.config = config['servo']
        self.usb_vendor_id = 0x1FFB  # Polulu
        self.usb_product_id = 0x0089  # Maestro 6
        self.device = None
        self.configuration = None
        self.isAutonomous = False
        self.speed_offset = 0
        self.steer_offset = 0
        self.__context, self.__subscriber = derp.util.subscriber(
            ["/tmp/derp_brain", "/tmp/derp_joystick"]
        )
        self.is_connected = self.__connect()
        if self.is_connected:
            self.reset()

    def reset(self):
        self.__send(self.config["speed_index"], 0, -1, 1)
        self.__send(self.config["steer_index"], 0, -1, 1)
        
    def __del__(self):
        if self.device:
            self.reset()

    def __connect(self):
        """ Re-initialize connection to USB servo """
        try:
            self.device = usb.core.find(idVendor=self.usb_vendor_id, idProduct=self.usb_product_id)
            self.configuration = self.device.get_active_configuration()
        except Exception as e:
            return False
        return True

    def __send(self, value, index, min_value, max_value):
        """ Actually send the message through USB to set the servo to the desired value """
        command = int((1500 + 500 * min(max(value, min_value), max_value)) * 4)
        try:
            self.device.ctrl_transfer(0x40, 0x85, command, index)
        except Exception as e:
            return False
        return True

    def run(self):
        """ Send the servo a specific value in [-1, 1] to move to """
        if not self.is_connected:
            print("servo: not connected")
            return False
        topic_bytes, message_bytes = self.__subscriber.recv_multipart()
        topic = topic_bytes.decode()
        message = derp.util.TOPICS[topic].from_bytes(message_bytes).as_builder()
        if topic == "controller":
            self.isAutonomous = message.isAutonomous
            self.speed_offset = message.speedOffset
            self.steer_offset = message.steerOffset
        elif topic == "action":
            if self.isAutonomous or message.isManual:
                self.__send(
                    message.speed + self.speed_offset,
                    self.config["speed_index"],
                    self.config["speed_min"],
                    self.config["speed_max"],
                )
                self.__send(
                    message.steer + self.steer_offset,
                    self.config["steer_index"],
                    self.config["steer_min"],
                    self.config["steer_max"],
                )


def run(config):
    servo = Servo(config)
    while servo.run():
        pass

