"""The Camera manages the camera interface and sends camera messages."""
import os
import re
import cv2
import time
import derp.util


class Camera:
    """The Camera manages the camera interface and sends camera messages."""

    def __init__(self, config):
        """The Camera manages the camera interface and sends camera messages."""
        self.config = config['camera']
        self.quality = [cv2.IMWRITE_JPEG_QUALITY, self.config['quality']]
        self.cap = None
        self.jpg = ''
        self.recv_timestamp = 0
        self.is_connected = self.__connect()
        self.__context, self.__publisher = derp.util.publisher("/tmp/derp_camera")

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        self.__publisher.close()
        self.__context.term()

    def __connect(self):
        if self.cap:
            self.cap.release()
            del self.cap
            self.cap = None
            time.sleep(1)
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
        print(gst)
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return bool(self.cap) and self.cap.isOpened()

    def publish_camera(self):
        message = derp.util.TOPICS['camera'].new_message(
            timeCreated=self.recv_timestamp,
            timePublished=derp.util.get_timestamp(),
            index=self.config['index'],
            jpg=self.jpg,
        )
        self.__publisher.send_multipart([b"camera", message.to_bytes()])
    
    def run(self):
        """Get and publish the camera frame"""
        if not self.is_connected:
            print("camera: not connected")
            return False
        ret, frame = self.cap.read()
        if not ret:
            print("camera: unable to read")
            return False
        self.recv_timestamp = derp.util.get_timestamp()
        self.jpg = cv2.imencode(".jpg", frame, self.quality)[1].tostring()
        self.publish_camera()
        return True

def run(config):
    """Run the camera in a loop"""
    camera = Camera(config)
    while camera.run():
        pass
