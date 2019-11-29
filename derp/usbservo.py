#!/usr/bin/env python3
import os
import usb.core
import usb.util
from derp.component import Component

class UsbServo(Component):
    """
    Interface through USB to the servo controller. At the moment the only
    supported capabilities are vague controls of the speed and steering. 
    """

    def __init__(self, config, state):

        super(UsbServo, self).__init__(config, state)
        self.device = None
        
        self.usb_vendor_id = 0x1ffb # Polulu
        self.usb_product_id = 0x0089  # maestro 6
        
        self.state_name = self.config['act_state']
        self.offset_name = 'offset_' + self.state_name
        self.use_offset_name = 'use_offset_' + self.state_name

        self.__connect()
        self.ready = True

    def __del__(self):
        self.__send(0) # make sure we tell the car to stop

    def __connect(self):
        """ Re-initialize connection to USB servo """
        try:
            self.device = usb.core.find(idVendor=self.usb_vendor_id,
                                        idProduct=self.usb_product_id)
            self.configuration = self.device.get_active_configuration() 
            self.ready = True
        except Exception as e:
            print("usbservo initialize:", e)
        
    def __send(self, value):
        """ Actually send the message through USB to set the servo to the desired value """
        # Limit command to known limits and convert to command
        value = min(value, self.config['max_value'])
        value = max(value, self.config['min_value'])
        command = int((1500 + 500 * value) * 4)
        try:
            self.device.ctrl_transfer(0x40, 0x85, command, self.config['index'])
            return True
        except Exception as e:
            print("usbservo send:", e)
            self.ready = False
            return False

    def act(self):
        """ Send the servo a specific value in [-1, 1] to move to """
        if not self.ready:
            self.__connect()

        value = self.state[self.state_name]
        if self.state[self.use_offset_name]:
            value += self.state[self.offset_name]
        if self.state.done():
            value = 0

        return self.__send(value)
