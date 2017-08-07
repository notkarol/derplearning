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
                 min_angle=-0.9,         # max angle to turn (left)
                 max_angle=0.9,          # max angle to turn (right)
    ):      
        """
        Interface through USB to the servo controller. At the moment the only
        supported capabilities are vague controls of the speed and steering. 
        """

        self.usb_vendor_id = usb_vendor_id
        self.usb_product_id = usb_product_id
        self.servo_speed_id = servo_speed_id
        self.servo_steer_id = servo_steer_id
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_speed = min_speed
        self.max_speed = max_speed
        
        # Initialize usb device
        self.device = usb.core.find(idVendor=self.usb_vendor_id,
                                    idProduct=self.usb_product_id)
        self.configuration = self.device.get_active_configuration()

        # Initialize state
        self.angle = 0
        self.speed = 0
        
        # Send command
        self.turn()
        self.move()

    def __del__(self):
        """
        Reset the car to its null state
        """
        self.move(0)
        self.turn(0)


    def move_faster(self,
                    amount=0.02): # amount to increase speed by
        """
        A wrapper around move that makes us go a bit faster
        """
        if self.speed == self.max_speed:
            return False
        self.speed += abs(amount)
        self.speed = min(self.speed, self.max_speed)
        self.move()
        return True

    
    def move_slower(self,
                    amount=0.02): # amount to decrease speed by
        """
        A wrapper around move that makes us go a bit slower
        """
        if self.speed <= self.min_speed:
            return False
        self.speed -= abs(amount)
        self.speed = max(self.speed, self.min_speed)
        self.move()
        return True


    def turn_right(self,
                   amount=0.02): # amount to turn right by
        """
        A wrapper around turn that makes us turn right a bit more
        """
        if self.angle >= self.max_angle:
            return False
        self.angle += abs(amount)
        self.angle = min(self.angle, self.max_angle)
        self.turn()
        return True


    def turn_left(self,
                  amount=0.02): # amount to turn left by
        """
        A wrapper around turn that makes us turn left a bit more
        """
        if self.angle == self.min_angle:
            return False
        self.angle -= abs(amount)
        self.angle = max(self.angle, self.min_angle)
        self.turn()
        return True
    
    
    def convert(self,
                intensity):  # normalized intensity value [-1, 1]
        """
        Convert an intensity value to the matching  Micro Maestro output value
        """
        intensity = max(intensity, -1)
        intensity = min(intensity, 1)
        return int((1500 + 500 * intensity) * 4)
        

    def move(self,
             speed=None): # speed values in [-1, 1]
        """
        Zero represents no movement. There is a dead zone around [-0.05, 0.05]
        Due to the stock ESC limitations, to go in reverse you have to send a
        negative command, a zero, and then a negative command again.
        """
        request_type = 0x40
        request = 0x85
        value = self.convert(self.speed if speed is None else speed)
        return self.device.ctrl_transfer(request_type, request,
                                         value, self.servo_speed_id)


    def turn(self,
             angle=None): # steering angle in [-1, 1]
        """
        Zero represents the wheels pointed forward. 
        +- 1 represent a turning radius or approximately 1 meter
        """
        request_type = 0x40
        request = 0x85
        value = self.convert(self.angle if angle is None else angle)
        return self.device.ctrl_transfer(request_type, request,
                                         value, self.servo_steer_id)
