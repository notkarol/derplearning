#!/usr/bin/env python3

import os
from derp.component import Component
from time import time
from Adafruit_BNO055 import BNO055

# The class that manages the IMU lives here

class BNO055(Component):

    def __init__(self, config):
        self.config = config

        self.bno = BNO055.BNO055(busnum=config['busnum'])
        if not self.bno.begin():
            raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')

        #Remap Axes to match camera's principle axes
        self.bno.set_axis_remap(x = AXIS_REMAP_Y,
                                y = AXIS_REMAP_Z,
                                z = AXIS_REMAP_X,
                                x_sign = AXIS_REMAP_POSITIVE,
                                y_sign = AXIS_REMAP_NEGATIVE,
                                z_sign = AXIS_REMAP_NEGATIVE)

        #Collect some nice status data that we can print as we do
        print("BNO055 status: %s self_test: %s error: %s" % self.bno.get_system_status())
        print("BNO055 sw: %s bl: %s accel: %s mag: %s gyro: %s" % self.bno.get_revision())        

    # Responsible for updating settings or acting upon the world
    def act(self, state):
        return True

    
    # Responsible for finding and connecting to the appropriate sensor[s]
    def discover(self):
        return False

    
    def scribe(self, folder):
        return True

    
    def sense(self, state):
        """ Read in sensor data """
        
        quaternion = self.bno.read_quaternion()
        euler = self.bno.read_euler()
        gravity = self.read_gravity()
        magneto = self.bno.read_magnetometer()
        gyro = self.bno.read_gyroscope()
        accel = self.bno.read_linear_acceleration()
        temp = self.bno.read_temp()
        timestamp = state['timestamp']

        #update state values:
        (state['quaternion_w'],
        state['quaternion_x'],
        state['quaternion_y'],
        state['quaternion_z']) = quaternion
        (state['euler_h'],
        state['euler_r'],
        state['euler_p']) =   euler
        (state['gravity_x'],
        state['gravity_y'],
        state['gravity_z']) = gravity
        (state['magneto_x'],
        state['magneto_y'],
        state['magneto_z']) = magneto
        (state['gyro_x'],
        state['gyro_y'],
        state['gyro_z']) = gyro
        (state['accel_x'],
        state['accel_y'],
        state['accel_z']) = accel
        state['temp'] = temp

        return True

    
    def write(self):
        return True
