"""
The bno055 is an IMU sensor. This class lets us communicate with it
in the derp way through the Adafruit BNO055 class.
"""
import Adafruit_BNO055.BNO055
import time
from derp.part import Part
import derp.util


class Imu(Part):
    """
    The bno055 is an IMU sensor. This class lets us communicate with it
    in the derp way through the Adafruit BNO055 class.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): The configuration file for the sensor.
        """
        super(Imu, self).__init__(config, "imu", [])
        self.busnum = self.config["busnum"] if "busnum" in self.config else -1
        self._bno = None
        self.calibration = None
        self.last_read_calibration = 0.0
        self.calibration_status = [0, 0, 0]
        self.angular_velocity = [0, 0, 0]
        self.magnetic_field = [0, 0, 0]
        self.linear_acceleration = [0, 0, 0]
        self.gravity = [0, 0, 0]
        self.orientation_quaternion = [0, 0, 0, 0]
        self.temperature = 0

    def __connect(self):
        """
        Are we connected to the BNO055 device through the provided
        vendor's object. If so, update the object and return true.
        Otherwise return False.
        """
        try:
            self._bno = Adafruit_BNO055.BNO055.BNO055(busnum=self.busnum)
        except PermissionError:
            print("imu: permission error")
            self._bno = None
            return
        except FileNotFoundError:
            print("imu: did you specify the right busnum in config?")
            self._bno = None
            return
        if not self._bno.begin():
            print("imu: unable to begin")
            self._bno = None
            return

        # Remap Axes to match camera's principle axes
        self._bno.set_axis_remap(
            x=Adafruit_BNO055.BNO055.AXIS_REMAP_Y,
            y=Adafruit_BNO055.BNO055.AXIS_REMAP_Z,
            z=Adafruit_BNO055.BNO055.AXIS_REMAP_X,
            x_sign=Adafruit_BNO055.BNO055.AXIS_REMAP_POSITIVE,
            y_sign=Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
            z_sign=Adafruit_BNO055.BNO055.AXIS_REMAP_NEGATIVE,
        )
        self.last_read_calibration = derp.util.get_timestamp()
        self.calibration = self._bno.get_calibration()

    def is_calibrated(self):
        for val in self.calibration_status:
            if val < 2:
                return False
        return True

    def run(self):
        """
        Reinitialize IMU if it's failed to get data at any point.
        Otherwise get data from the IMU to update state variable.
        """
        if self._bno == None:
            self.__connect()
        self._timestamp = derp.util.get_timestamp()
        try:
            if self._timestamp - self.last_read_calibration > 1e6:
                self.last_read_calibration = derp.util.get_timestamp()
                self.calibration = self._bno.get_calibration()
            self.calibration_status = self._bno.get_calibration_status()
            self.angular_velocity = self._bno.read_gyroscope()
            self.magnetic_field = self._bno.read_magnetometer()
            self.linear_acceleration = self._bno.read_linear_acceleration()
            self.gravity = self._bno.read_gravity()
            self.orientation_quaternion = self._bno.read_quaternion()
            self.temperature = self._bno.read_temp()
        except OSError:
            print("IMU FAILED", self._timestamp)
        finally:
            self.publish(
                "imu",
                index=self.busnum,
                isCalibrated=self.is_calibrated(),
                angularVelocity=self.angular_velocity,
                magneticField=self.magnetic_field,
                linearAcceleration=self.linear_acceleration,
                gravity=self.gravity,
                orientationQuaternion=self.orientation_quaternion,
                temperature=self.temperature,
            )
        derp.util.sleep_hertz(self._timestamp, 100)
        return True
