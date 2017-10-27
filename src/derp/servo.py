#!/usr/bin/env python3

import time
import usb.core
import usb.util

class Servo:

    def __init__(self,
                 usb_vendor_id=0x1ffb,   # polulu
                 usb_product_id=0x0089,  # maestro 6
                 servo_speed_id=1,       # ordered index number of speed
                 servo_steer_id=0,       # ordered index number of steering
                 min_speed=-0.5,         # slowest to go (or fastest in reverse)
                 max_speed=0.5,          # fastest to go
                 min_steer=-0.9,         # max steer to turn (left)
                 max_steer=0.9,          # max steer to turn (right)
                 turn_offset=0.0         # how much to adjust 0 turn to drive straight
    ):      
        """
        Interface through USB to the servo controller. At the moment the only
        supported capabilities are vague controls of the speed and steering. 
        """

        self.usb_vendor_id = usb_vendor_id
        self.usb_product_id = usb_product_id
        self.servo_speed_id = servo_speed_id
        self.servo_steer_id = servo_steer_id
        self.min_steer = min_steer
        self.max_steer = max_steer
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.turn_offset = turn_offset

        # Initialize usb device
        self.configuration = None
        self.device = usb.core.find(idVendor=self.usb_vendor_id,
                                    idProduct=self.usb_product_id)
        if self.device is not None:
            self.configuration = self.device.get_active_configuration() 
        
        # Send command
        self.move(0)
        self.turn(0)

        
    def __del__(self):
        """
        Reset the car to its null state
        """
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
        

    def move(self, speed):
        """
        Zero represents no movement. There is a dead zone around [-0.05, 0.05]
        Due to the stock ESC limitations, to go in reverse you have to send a
        negative command, a zero, and then a negative command again.
        """

        # If we're not connected to a device then just act dead
        if self.device is None:
            return None

        # Make sure we have a speed
        self.speed = speed
        self.speed = min(self.max_speed, self.speed)
        self.speed = max(self.min_speed, self.speed)
            
        # Send request to servo
        return self.device.ctrl_transfer(0x40, 0x85, self.convert(self.speed), self.servo_speed_id)


    def turn(self, steer):
        """
        Zero represents the wheels pointed forward. 
        +- 1 represent a turning radius of approximately 1 meter
        """

        # If we're not connected to a device then just act dead
        if self.device is None:
            return None

        # Make sure we have an steer
        self.steer = steer + self.turn_offset
        self.steer = min(self.max_steer, self.steer)
        self.steer = max(self.min_steer, self.steer)
        
        # Send request to servo
        return self.device.ctrl_transfer(0x40, 0x85, self.convert(self.steer), self.servo_steer_id)
