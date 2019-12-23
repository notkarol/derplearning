"""
The bno055 is an IMU sensor. This class lets us communicate with it
in the derp way through the Adafruit BNO055 class.
"""
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
        self.config = config['imu']
        self.bno = None
        self.quaternion = None
        self.gravity = None
        self.magnetometer = None
        self.gyroscope = None
        self.acceleropmeter = None
        self.calibration_status = None
        self.calibration = None
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

    def run(self):
        """
        Reinitialize IMU if it's failed to get data at any point.
        Otherwise get data from the IMU to update state variable.
        """
        calibration_status = self.bno.get_calibration_status()
        msg = derp.util.TOPICS['imu'].new_message(
            timeCreated=derp.util.get_timestamp(),
            calibrationGyroscope=calibration_status[1],
            calibrationAccelerometer=calibration_status[2],
            calibrationMagnetometer=calibration_status[3],
            accelerometer=self.bno.read_linear_acceleration(),
            calibration=self.bno.calibration(),
            gravity=self.bno.read_gravity(),
            gyroscope=self.bno.read_gyroscope(),
            magnetometer=self.bno.read_magnetometer(),
            quaternion=self.bno.read_quaternion(),
        )
        msg.timePublished = derp.util.get_timestamp()
        


def run(config):
    """Run the IMU in a loop"""
    imu = BNO055(config)
    while True:
        imu.run()
