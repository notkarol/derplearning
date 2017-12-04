#!/usr/bin/env python3

import os
from derp.component import Component
from time import time
import Adafruit_BNO055.BNO055


class BNO055(Component):
    """
    IMU processing class
    """

    def __init__(self, config, full_config):
        super(BNO055, self).__init__(config, full_config)

        # Connect to the imu and then initialize it
        self.bno = Adafruit_BNO055.BNO055.BNO055(busnum=self.config['busnum'])
        if not self.bno.begin():
            return

        # Remap Axes to match camera's principle axes
        self.bno.set_axis_remap(x = Adafruit_BNO055.BNO055.AXIS_REMAP_Y,
                                y = Adafruit_BNO055.BNO055.AXIS_REMAP_Z,
                                z = Adafruit_BNO055.BNO055.AXIS_REMAP_X,
                                x_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_POSITIVE,
                                y_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
                                z_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE)

        # Collect some nice status data that we can print as we do
        print("BNO055 status: %s self_test: %s error: %s" % self.bno.get_system_status())
        print("BNO055 sw: %s bl: %s accel: %s mag: %s gyro: %s" % self.bno.get_revision())        
        self.ready = True

    
    def sense(self, state):

        # Read in sensor data
        quaternion = self.bno.read_quaternion()
        euler = self.bno.read_euler()
        gravity = self.bno.read_gravity()
        magneto = self.bno.read_magnetometer()
        gyro = self.bno.read_gyroscope()
        accel = self.bno.read_linear_acceleration()
        temp = self.bno.read_temp()
        timestamp = state['timestamp']

        # Update state
        state.update_multipart('quaternion', 'wxyz', quaternion)
        state.update_multipart('euler', 'hrp', euler)
        state.update_multipart('gravity', 'xyz', gravity)
        state.update_multipart('magneto', 'xyz', magneto)
        state.update_multipart('gyro', 'xyz', gyro)
        state.update_multipart('accel', 'xyz', accel)
        state['temp'] = temp

        return True
