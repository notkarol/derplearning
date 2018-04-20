#!/usr/bin/env python3

import os
from time import time
import yaml

import Adafruit_BNO055.BNO055

from derp.component import Component

class BNO055(Component):

    def __init__(self, config, full_config, state):
        super(BNO055, self).__init__(config, full_config, state)

        self.bno = Adafruit_BNO055.BNO055.BNO055(busnum=self.config['busnum'])
        self.ready = self.bno.begin()
        if not self.ready:
            return

        # Remap Axes to match camera's principle axes
        self.bno.set_axis_remap(x = Adafruit_BNO055.BNO055.AXIS_REMAP_Y,
                                y = Adafruit_BNO055.BNO055.AXIS_REMAP_Z,
                                z = Adafruit_BNO055.BNO055.AXIS_REMAP_X,
                                x_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_POSITIVE,
                                y_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
                                z_sign = Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE)

        self.calibration_saved = False
        if os.path.exists(self.config['calibration_path']):
            with open(self.config['calibration_path']) as f:
                calibration = yaml.load(f)
            self.bno.set_calibration(calibration)
            self.calibration_saved = self.is_calibrated()

        print("BNO055 status: %s self_test: %s error: %s" % self.bno.get_system_status() )
        print("BNO055 sw: %s bl: %s accel: %s mag: %s gyro: %s" % self.bno.get_revision())
        
    def is_calibrated(self):
        return self.bno.get_calibration_status() == (3, 3, 3, 3)

    def print_calibration_status(self):
        print("System: %i | Gyro: %i | Accel: %i | Mag: %i" % self.bno.get_calibration_status())
    
    def sense(self):
        self.state.update_multipart('quaternion', 'wxyz', self.bno.read_quaternion())
        self.state.update_multipart('euler', 'hrp', self.bno.read_euler())
        self.state.update_multipart('gravity', 'xyz', self.bno.read_gravity())
        self.state.update_multipart('magneto', 'xyz', self.bno.read_magnetometer())
        self.state.update_multipart('gyro', 'xyz', self.bno.read_gyroscope())
        self.state.update_multipart('accel', 'xyz', self.bno.read_linear_acceleration())
        self.state.update_multipart('calibration', ('system', 'gyro', 'accel', 'mag'),
                                    self.bno.get_calibration_status())
        self.state['temp'] = self.bno.read_temp()
        return True

    def record(self):
        """
        Store 22 bytes of calibration data to a pre-set file, as according to the config.
        """
        if self.calibration_saved:
            return True
        calibration = self.bno.get_calibration()
        if sum(calibration) != 12:
            return False
        with open(self.config['calibration_path'], 'w') as yaml_file:
            yaml.dump(calibration, yaml_file, default_flow_style=False)
        self.calibration_saved = True
        return True

