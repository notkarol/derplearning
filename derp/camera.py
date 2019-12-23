"""The Camera manages the camera interface and sends camera messages."""
import os
import re
import cv2
import derp.util


class Camera:
    """The Camera manages the camera interface and sends camera messages."""

    def __init__(self, config):
        """The Camera manages the camera interface and sends camera messages."""
        self.config = config['camera']
        self.cap = None
        self.__connect()
        self.width = int(self.config["width"] * self.config["resize"] + 0.5)
        self.height = int(self.config["height"] * self.config["resize"] + 0.5)
        self.__context, self.__publisher = derp.util.publisher("/tmp/derp_camera")

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        self.__publisher.close()
        self.__context.term()

    def __connect(self):
        if self.cap:
            del self.cap
            self.cap = None
        if self.config["index"] is None:
            devices = [
                int(f[-1]) for f in sorted(os.listdir("/dev")) if re.match(r"^video[0-9]", f)
            ]
            if len(devices) == 0:
                return False
            self.index = devices[-1]
        else:
            self.index = self.config["index"]

        # Connect to camera, exit if we can't
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap or not self.cap.isOpened():
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        return True

    def run(self):
        """Get and publish the camera frame"""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.__connect()
            return
        msg = derp.util.TOPICS['camera'].new_message(
            timeCreated=derp.util.get_timestamp(),
            yaw=self.config["yaw"],
            pitch=self.config["pitch"],
            roll=self.config["roll"],
            x=self.config["x"],
            y=self.config["y"],
            z=self.config["z"],
            height=self.config["height"],
            width=self.config["width"],
            depth=self.config["depth"],
            hfov=self.config["hfov"],
            vfov=self.config["vfov"],
            fps=self.config["fps"],
        )
        frame = derp.util.resize(frame, (self.width, self.height))
        msg.jpg = cv2.imencode(".jpg", frame)[1].tostring()
        msg.timePublished = derp.util.get_timestamp()
        self.__publisher.send_multipart([b"camera", msg.to_bytes()])


def run(config):
    """Run the camera in a loop"""
    camera = Camera(config)
    while True:
        camera.run()
