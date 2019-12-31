"""
The bno055 is an IMU sensor. This class lets us communicate with it
in the derp way through the Adafruit BNO055 class.
"""
import Adafruit_BNO055.BNO055
import time
import derp.util

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
        self.calibration = None
        self.is_connected = self.__connect()
        self.__context, self.__publisher = derp.util.publisher("/tmp/derp_imu")

    def __del__(self):
        self.__publisher.close()
        self.__context.term()
        
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
        self.calibration = self.bno.get_calibration()
        return True

    def run(self):
        """
        Reinitialize IMU if it's failed to get data at any point.
        Otherwise get data from the IMU to update state variable.
        """
        recv_time = derp.util.get_timestamp()
        calibration_status = self.bno.get_calibration_status()
        accelerometer = self.bno.read_linear_acceleration()
        gravity = self.bno.read_gravity()
        gyroscope = self.bno.read_gyroscope()
        magnetometer = self.bno.read_magnetometer()
        quaternion = self.bno.read_quaternion()
        message = derp.util.TOPICS['imu'].new_message(
            timeCreated=recv_time,
            timePublished=derp.util.get_timestamp(),
            calibrationGyroscope=calibration_status[1],
            calibrationAccelerometer=calibration_status[2],
            calibrationMagnetometer=calibration_status[3],
            accelerometer=accelerometer,
            calibration=bytes(self.calibration),
            gravity=gravity,
            gyroscope=gyroscope,
            magnetometer=magnetometer,
            quaternion=quaternion,
        )
        self.__publisher.send_multipart([b"imu", message.to_bytes()])
        derp.util.sleep_hertz(recv_time, 100)
        return True

def run(config):
    """Run the IMU in a loop"""
    imu = BNO055(config)
    while imu.run():
        pass
