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

        # Connect to the imu and then initialize it using adafruit class
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

        #IMU calibration status and checks
        print("BNO055 sytem calibration status: %s gyro: %s accel: %s mag: %s" % self.cal_report())
        self.sys_cal = self.cal_report()[0]
    
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

        if not self.sys_cal == 3:
            print("BNO055 sytem calibration status: %s gyro: %s accel: %s mag: %s" % self.cal_report(), end="/r")


        return True


    #reports the calibration status as a tuple: (sys, gyro, accel, mag)
    def cal_report(self):
        return self.bno.get_calibration_status()


    """Return the sensor's calibration data and return it as an array of
    22 bytes. Can be saved and then reloaded with the set_calibration function
    to quickly calibrate from a previously calculated set of calibration data.
    """
    def read_calibration(self):
        return self.bno.get_calibration()


    def load_saved_calibration(self, cal_data):

        self.set_calibration(cal_data)

        return True
