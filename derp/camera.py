"""The Camera manages the camera interface and sends camera messages."""
import cv2
import time
from derp.part import Part
import derp.util


class Camera(Part):
    """The Camera manages the camera interface and sends camera messages."""

    def __init__(self, config):
        """The Camera manages the camera interface and sends camera messages."""
        super(Camera, self).__init__(config, "camera", [])
        self._cap = None
        self.size = (self._config["width"], self._config["height"])
        self._frame = None
        self.__connect()

    def __del__(self):
        super(Camera, self).__del__()
        if self._cap is not None:
            self._cap.release()

    def __connect(self):
        if self._cap is not None:
            self._cap.release()
            del self._cap
            self._cap = None
            time.sleep(1)

        device = "device=/dev/video%i" % self._config["index"]
        gst = None
        if self._config["mode"] == "video":
            gst = (
                "v4l2src %s"
                " ! video/x-raw,format=YUY2,width=%i,height=%i,framerate=%i/1 "
                " ! videoconvert ! appsink"
                % (device, self._config["width"], self._config["height"], self._config["fps"])
            )
        elif self._config["mode"] == "image":
            gst = (
                "v4l2src %s"
                " ! image/jpeg,width=%i,height=%i,framerate=%i/1"
                " ! jpegparse ! jpegdec ! videoconvert ! appsink"
                % (device, self._config["width"], self._config["height"], self._config["fps"])
            )
        elif self._config["mode"] == "csi":
            gst = (
                "nvarguscamerasrc sensor-id=%i"
                " ! video/x-raw(memory:NVMM),width=%i,height=%i,framerate=(fraction)%i/1,format=(string)NV12"
                " ! nvvidconv flip-method=0"
                " ! video/x-raw,width=%i,height=%i,format=BGRx"
                " ! videoconvert ! appsink"
                % (
                    self._config["index"],
                    self._config["capture_width"],
                    self._config["capture_height"],
                    self._config["fps"],
                    self._config["width"],
                    self._config["height"],
                )
            )

        print(gst)
        if gst is not None:
            self._cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    def read(self):
        """ Read the camera image if possible, and if so update the timestamp we received data """
        ret, self._frame = self._cap.read()
        self._timestamp = derp.util.get_timestamp()
        return ret

    def run(self):
        """Get and publish the camera frame"""
        if not self.read():
            return False
        self.publish(
            "camera",
            index=self._config["index"],
            jpg=derp.util.encode_jpg(self._frame, self._config["quality"]),
        )
        return True
