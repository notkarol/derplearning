#!/usr/bin/env python3

from time import time
import usb.core
import usb.util
from derp.component import Component

class UsbServo(Component):

    def __init__(self, config, name):
        """
        Interface through USB to the servo controller. At the moment the only
        supported capabilities are vague controls of the speed and steering. 
        """

        if 'usb_vendor_id' in self.config[name]:
            self.usb_vendor_id = self.config[name]['usb_vendor_id']
        else:
            self.usb_vendor_id = 0x1ffb # Polulu

        if 'usb_product_id' in self.config[name]:
            self.usb_product_id = self.config[name]['usb_product_id']
        else:
            self.usb_product_id = 0x0089  # maestro 6

        if 'min_value' in self.config[name]:
            self.min_value = self.config[name]['min_value']
        else:
            self.min_value = -0.95
        if 'max_value' in self.config[name]:
            self.max_value = self.config[name]['max_value']
        else:
            self.max_value = 0.95

            
        self.servo_id = self.config[name]['index']

        
    def __del__(self):
        pass


    def act(self, state):

        if self.device is None:
            return False
        
        timestamp = int(time() * 1E6)
        value = state[name]
        value = min(value, self.max_value)
        value = max(value, self.min_value)
        value = int((1500 + 500 * intensity) * 4)
        
        self.out_buffer.append((time, value))
        
        return self.device.ctrl_transfer(0x40, 0x85, value, self.servo_id)
                                

    def discover(self):

        self.configuration = None
        self.device = usb.core.find(idVendor=self.usb_vendor_id,
                                    idProduct=self.usb_product_id)
        if self.device is None:
            return False
        
        self.configuration = self.device.get_active_configuration() 
        return True

    
    def folder(self, folder):
        if self.out_csv is not None:
            self.out.csv.close()
        self.out_csv_path = os.path.join(folder, "%s.csv" % self.name)
        self.out_csv = open(self.name, 'w')   

    
    def sense(self):
        return True

    
    def write(self):
        if self.out_csv_fp is None:
            return False

        for row in self.out_buffer:
            self.out_csv_fp.write(",".join(row) + "\n")
        self.out_csv_fp.flush()
        self.out_buffer = []
        return True       
