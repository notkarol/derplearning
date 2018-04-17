#!/usr/bin/env python3

import os
from derp.component import Component
from time import time
import Adafruit_BNO055.BNO055
import yaml


class BNO055(Component):
    """
    IMU processing class
    """

    def __init__(self, config, full_config, state):
        super(BNO055, self).__init__(config, full_config, state)

        # Connect to the imu and then initialize it using adafruit class
        self.bno = Adafruit_BNO055.BNO055.BNO055(busnum=self.config['busnum'])
        if not self.bno.begin():
            return
        self.calibration_status = self.bno.get_calibration_status()

        # Remap Axes to match camera's principle axes
        self.bno.set_axis_remap(x = Adafruit_BNO055.BNO055.AXIS_REMAP_Y,
                                y = Adafruit_BNO055.BNO055.AXIS_REMAP_Z,
                                z = Adafruit_BNO055.BNO055.AXIS_REMAP_X,
                                x_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_POSITIVE,
                                y_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
                                z_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE)

        # Collect some nice status data that we can print as we do
        print("BNO055 status: %s self_test: %s error: %s" % self.bno.get_system_status() )
        print("BNO055 sw: %s bl: %s accel: %s mag: %s gyro: %s" % self.bno.get_revision())
        
        #check calibration level and attempt to load calibration data from file
        with open(self.config['calibration_path']) as f:
            self.cal_config = yaml.load(f)
        if ( not (3, 3, 3, 3) == self.calibration_status ) :
            if self.load_saved_calibration(self.cal_config['calibrations']):
                print("Loaded calibrations with status: %s" % self.cal_config['status'])
            else: print("Failed to load saved calibrations.")
            self.calibration_status = self.bno.get_calibration_status()
        print("BNO055 sytem calibration status: %s gyro: %s accel: %s mag: %s" %
              self.calibration_status, end="\r")
        self.calibration_flag = (self.calibration_status == (3, 3, 3, 3))
        self.ready = True
    
    def sense(self):

        # Read in sensor data
        quaternion = self.bno.read_quaternion()
        euler = self.bno.read_euler()
        gravity = self.bno.read_gravity()
        magneto = self.bno.read_magnetometer()
        gyro = self.bno.read_gyroscope()
        accel = self.bno.read_linear_acceleration()
        temp = self.bno.read_temp()
        calibration_status = self.bno.get_calibration_status()
        timestamp = self.state['timestamp']

        # Update state
        self.state.update_multipart('quaternion', 'wxyz', quaternion)
        self.state.update_multipart('euler', 'hrp', euler)
        self.state.update_multipart('gravity', 'xyz', gravity)
        self.state.update_multipart('magneto', 'xyz', magneto)
        self.state.update_multipart('gyro', 'xyz', gyro)
        self.state.update_multipart('accel', 'xyz', accel)
        self.state['temp'] = temp
        self.state['warn'] |= not self.calibration_flag
        if self.calibration_flag:
            self.save_calibration()
        self.calibration_flag = (calibration_status == (3, 3, 3, 3) )
        return True


    def cal_report(self):
        """
        Reports the calibration status as a tuple: (sys, gyro, accel, mag)
        """
        return self.bno.get_calibration() 


    #stores good calibrations settings for future use.
    def save_calibration(self):
        """
        Return the sensor's calibration data and return it as an array of
        22 bytes. Can be saved and then reloaded with the set_calibration function
        to quickly calibrate from a previously calculated set of calibration data.
        """
        calibration = {
            'status' : list(self.bno.get_calibration_status()),
            'calibrations' : self.bno.get_calibration() }

        with open(self.config['calibration_path'], 'w') as yaml_file:
            yaml.dump(calibration, yaml_file, default_flow_style=False)

        return True


    def load_saved_calibration(self, calibration_data):

        self.bno.set_calibration(calibration_data)

        return True
