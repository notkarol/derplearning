#!/usr/bin/env python3

import time
import usb.core
import usb.util

class Servo:

    def __init__(self, vendor_id=None, product_id=None):
        self.vendor_id = 0x1ffb # pololu
        self.product_id = 0x0089 # maestro 6

        # Handle arguments
        if vendor_id is not None:
            self.vendor_id = vendor_id
        if product_id is not None:
            self.product_id = product_id

        # Initialize usb device
        self.device = usb.core.find(idVendor=self.vendor_id,
                                    idProduct=self.product_id)
        self.configuration = self.device.get_active_configuration()

        # Identify servos
        self.move_id = 1
        self.turn_id = 0
        
        
    def convert(self, intensity):
        intensity = max(intensity, -1)
        intensity = min(intensity, 1)
        return int((1500 + 500 * intensity) * 4)
        

    def move(self, speed=0):
        request_type = 0x40
        request = 0x85
        value = self.convert(speed)
        return self.device.ctrl_transfer(request_type, request,
                                         value, self.move_id)
        
    def turn(self, angle=0):
        request_type = 0x40
        request = 0x85
        value = self.convert(angle)
        return self.device.ctrl_transfer(request_type, request,
                                         value, self.turn_id)
