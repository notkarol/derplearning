"""
The bno055 is an IMU sensor. This class lets us communicate with it
in the derp way through the Adafruit BNO055 class.
"""
import pathlib
import yaml
import capnp
import messages_capnp
import Adafruit_BNO055.BNO055


class BNO055:
    """
    The bno055 is an IMU sensor. This class lets us communicate with it
    in the derp way through the Adafruit BNO055 class.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): The configuration file for the sensor.
        """
        self.config = config
        self.bno = None
        self.__connect()

    def __connect(self):
        """
        Are we connected to the BNO055 device through the provided
        vendor's object. If so, update the object and return true.
        Otherwise return False.
        """
        self.bno = Adafruit_BNO055.BNO055.BNO055(busnum=self.config["busnum"])
        if not self.bno.begin():
            return False

        # Remap Axes to match camera's principle axes
        self.bno.set_axis_remap(
            x=Adafruit_BNO055.BNO055.AXIS_REMAP_Y,
            y=Adafruit_BNO055.BNO055.AXIS_REMAP_Z,
            z=Adafruit_BNO055.BNO055.AXIS_REMAP_X,
            x_sign=Adafruit_BNO055.BNO055.AXIS_REMAP_POSITIVE,
            y_sign=Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
            z_sign=Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
        )
        return True

    def create_imu_message(self):
        msg = messages_capnp.Camera.new_message(
            timestampCreated=derp.util.get_timestamp(),
            calibrationGyroscope=self.calibration_status[1],
            calibrationAccelerometer=self.calibration_status[2],
            calibrationMagnetometer=self.calibration_status[3],
            accelerometer=self.accelerometer,
            calibration=self.calibration,
            gravity=self.gravity,
            gyroscope=self.gyroscope,
            magnetometer=self.magnetometer,
            quaternion=self.quaternion,
        )
        return msg

    def run(self):
        """
        Reinitialize IMU if it's failed to get data at any point. 
        Otherwise get data from the IMU to update state variable.
        """
        self.temp = self.bno.read_temp()
        self.quaterion = self.bno.read_quaternion()
        self.gravity = self.bno.read_gravity()
        self.magnetometer = self.bno.read_magnetometer()
        self.gyroscope = self.bno.read_gyroscope()
        self.accelerometer = self.bno.read_linear_acceleration()
        self.calibration_status = self.bno.get_calibration_status()
        self.calibration = self.bno.calibration()
        message = self.create_imu_message()


def run(config):
    imu = BNO055(config)
    while True:
        imu.run()
