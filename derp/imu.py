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
        self.busnum = self.config['busnum'] if 'busnum' in self.config else -1
        self.bno = None
        self.calibration = None
        self.last_read_calibration = 0.0
        self.recv_time = 0.0
        self.is_connected = self.__connect()
        self.calibration_status = [0, 0, 0]
        self.angular_velocity = [0, 0, 0]
        self.magnetic_field = [0, 0, 0]
        self.linear_acceleration = [0, 0, 0]
        self.gravity = [0, 0, 0]
        self.orientation_quaternion = [0, 0, 0, 0]
        self.temperature = 0
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
        try:
            self.bno = Adafruit_BNO055.BNO055.BNO055(busnum=self.busnum)
        except PermissionError:
            print("imu: permission error")
            return False
        except FileNotFoundError:
            print("imu: did you specify the right busnum in config?")
            return False
        if not self.bno.begin():
            print("imu: unable to begin")
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
        self.last_read_calibration = derp.util.get_timestamp()
        self.calibration = self.bno.get_calibration()
        return True

    def publish_imu(self):
        message = derp.util.TOPICS['imu'].new_message(
            timeCreated=self.recv_time,
            timePublished=derp.util.get_timestamp(),
            index=self.busnum,
            isCalibrated=self.is_calibrated(),
            angularVelocity=self.angular_velocity,
            magneticField=self.magnetic_field,
            linearAcceleration=self.linear_acceleration,
            gravity=self.gravity,
            orientationQuaternion=self.orientation_quaternion,
            temperature=self.temperature,
        )
        self.__publisher.send_multipart([b"imu", message.to_bytes()])

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
        self.recv_time = derp.util.get_timestamp()
        if not self.is_connected:
            return False
        if self.recv_time - self.last_read_calibration > 1E6:
            self.last_read_calibration = derp.util.get_timestamp()
            self.calibration = self.bno.get_calibration()
        self.calibration_status = self.bno.get_calibration_status()
        self.angular_velocity = self.bno.read_gyroscope()
        self.magnetic_field = self.bno.read_magnetometer()
        self.linear_acceleration = self.bno.read_linear_acceleration()
        self.gravity = self.bno.read_gravity()
        self.orientation_quaternion = self.bno.read_quaternion()
        self.temperature = self.bno.read_temperature()
        self.publish_imu()
        derp.util.sleep_hertz(self.recv_time, 100)
        return True

def loop(config):
    """Run the IMU in a loop"""
    imu = BNO055(config)
    while imu.run():
        pass
    time.sleep(1)
    imu.publish_imu()
    time.sleep(1)
