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
        self.quality = [cv2.IMWRITE_JPEG_QUALITY, self.config['quality']]
        self.cap = None
        self.ready = self.__connect()
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
        if self.config['index'] is None:
            device = 'autovideosrc'
        else:
            device ='device=/dev/video%i' % self.config['index']
        width = self.config['width']
        height = self.config['height']
        fps = self.config['fps']

        if self.config['mode'] == 'video':
            gst = ('v4l2src %s ! video/x-raw,format=YUY2,width=%i,height=%i,framerate=%i/1 '
                   '! videoconvert ! appsink' % (device, width, height, fps))
        elif self.config['mode'] == 'image':
            gst = ('v4l2src %s ! image/jpeg,width=%i,height=%i,framerate=%i/1 '
                   '! jpegparse ! jpegdec ! videoconvert ! appsink' % (device, width, height, fps))
        else:
            return False
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return bool(self.cap) and self.cap.isOpened()

    def run(self):
        """Get and publish the camera frame"""
        ret, frame = self.cap.read()
        recv_timestamp = derp.util.get_timestamp()
        if not ret:
            self.ready = self.__connect()
            return
        jpg = cv2.imencode(".jpg", frame, self.quality)[1].tostring()
        msg = derp.util.TOPICS['camera'].new_message(
            timeCreated=recv_timestamp,
            timePublished=derp.util.get_timestamp(),
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
            jpg=jpg,
        )
        self.__publisher.send_multipart([b"camera", msg.to_bytes()])


def run(config):
    """Run the camera in a loop"""
    camera = Camera(config)
    while True:
        camera.run()
