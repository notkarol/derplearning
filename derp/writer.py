"""
The Camera component manages the camera interface.
"""

import derp.util
import derp.camera
import capnp
import camera_capnp
import input_capnp


class Writer:
    """
    The Camera component manages the camera interface.
    """

    def __init__(self, config):
        self.config = config
        self.__context, self.__subscriber = derp.util.subscriber(self.config['name'])
        self.files = {'camera': open('camera.bin',  'w+b'),
                      'input': open('input.bin',  'w+b')}

    def __del__(self):
        self.__subscriber.close()
        self.__context.term()

    def run(self):
        while True:
            frame = self.read()
            if frame is None:
                self.__connect()
                continue
            frame = derp.util.resize(frame, (self.width, self.height))
            msg = self.message(frame)
            topic, message = self.subscriber.recv_multipart()
            if topic == 'camera':
                camera_capnp.Camera.from_bytes(message).write(self.files[topic])
            elif topic == 'input':
                camera_capnp.Input.from_bytes(message).write(self.files[topic])
