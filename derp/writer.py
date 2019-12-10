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
                                                                  '/tmp/derp_brain',
                                                                  '/tmp/derp_joystick',
                                                                  '/tmp/derp_keyboard'])
        self.files = {}
        self.run()

    def __del__(self):
        self.__subscriber.close()
        self.__context.term()

    def run(self):
        topic_bytes, message_bytes = self.__subscriber.recv_multipart()
        topic = topic_bytes.decode()
        message = self.classes[topic].from_bytes(message_bytes).as_builder()

        # Create folder or delete folder
        if topic == 'state':
            if message.record and not self.files:
                folder = derp.util.create_record_folder()
                print(folder)
                self.files = {name: open('%s/%s.bin' % (folder, name), 'w+b')
                              for name in self.classes}
            elif not message.record and self.files:
                for name in self.files:
                    self.files[name].close()
                self.files = {}

        if self.files:
            message.timestampWritten = derp.util.get_timestamp()
            message.write(self.files[topic])

def run(config):
    writer = Writer(config)
    while True:
        writer.run()
