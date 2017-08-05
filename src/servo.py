#!/usr/bin/env python3

import time
import usb.core
import usb.util

class Servo:

    def __init__(self,
                 usb_vendor_id=0x1ffb,   # polulu
                 usb_product_id=0x0089,  # maestro 6
                 servo_speed_id=1,       # ordered index number of speed
                 servo_steer_id=0):      # ordered index number of steering
        """
        Interface through USB to the servo controller. At the moment the only
        supported capabilities are vague controls of the speed and steering. 
        """

        self.usb_vendor_id = usb_vendor_id
        self.usb_product_id = usb_product_id
        self.servo_speed_id = servo_speed_id
        self.servo_steer_id = servo_steer_id

        # Initialize usb device
        self.device = usb.core.find(idVendor=self.usb_vendor_id,
                                    idProduct=self.usb_product_id)
        self.configuration = self.device.get_active_configuration()

    def __del__(self):
        self.move(0)
        self.turn(0)

    def convert(self,
                intensity):  # normalized intensity value [-1, 1]
        """
        Convert an intensity value to the matching  Micro Maestro output value
        """
        intensity = max(intensity, -1)
        intensity = min(intensity, 1)
        return int((1500 + 500 * intensity) * 4)
        

    def move(self,
             speed=0): # speed values in [-1, 1]
        """
        Zero represents no movement. There is a dead zone around [-0.05, 0.05]
        Due to the stock ESC limitations, to go in reverse you have to send a
        negative command, a zero, and then a negative command again.
        """
        request_type = 0x40
        request = 0x85
        value = self.convert(speed)
        return self.device.ctrl_transfer(request_type, request,
                                         value, self.servo_speed_id)


    def turn(self,
             angle=0): # steering angle in [-1, 1]
        """
        Zero represents the wheels pointed forward. 
        +- 1 represent a turning radius or approximately 1 meter
        """
        request_type = 0x40
        request = 0x85
        value = self.convert(angle)
        return self.device.ctrl_transfer(request_type, request,
                                         value, self.servo_steer_id)
