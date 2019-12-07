"""
The Camera component manages the camera interface.
"""

import derp.util
import derp.camera
import capnp
import messages_capnp
import zmq

class Writer:
    """
    The Camera component manages the camera interface.
    """

    def __init__(self, config):
        self.config = config
        self.__context, self.__subscriber = derp.util.subscriber(['/tmp/derp_camera',
                                                                  '/tmp/derp_imu',
                                                                  '/tmp/derp_keyboard'])
        self.files = {'camera': open('camera.bin',  'w+b'),
                      'input': open('input.bin',  'w+b')}
        self.counter = 0
        self.run()

    def __del__(self):
        self.__subscriber.close()
        self.__context.term()

    def run(self):
        topic, message = self.__subscriber.recv_multipart()
        topic = topic.decode()
        if topic == 'camera':
            msg = messages_capnp.Camera.from_bytes(message)
        elif topic == 'control':
            msg = messages_capnp.Control.from_bytes(message)
        elif topic == 'state':
            msg = messages_capnp.Control.from_bytes(message)
        else:
            print("Skipping", topic)
        msg.timestampWritten = derp.util.get_timestamp()
        msg.as_builder().write(self.files[topic])
        self.counter += 1
        print(topic, self.counter)

def run(config):
    writer = Writer(config)
    while True:
        writer.run()
