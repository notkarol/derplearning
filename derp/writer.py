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
        self.classes = {'camera': messages_capnp.Camera,
                        'state': messages_capnp.State,
                        'control': messages_capnp.Control,
                        'imu': messages_capnp.Imu}
        self.files = {name: open('%s.bin' % name,  'w+b') for name in self.classes}
        self.run()

    def __del__(self):
        self.__subscriber.close()
        self.__context.term()

    def run(self):
        topic_bytes, message_bytes = self.__subscriber.recv_multipart()
        topic = topic_bytes.decode()
        message = self.classes[topic].from_bytes(message_bytes).as_builder()
        message.timestampWritten = derp.util.get_timestamp()
        message.write(self.files[topic])
        print(topic)

def run(config):
    writer = Writer(config)
    while True:
        writer.run()
