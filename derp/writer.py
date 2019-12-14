"""The disk writer class that records all derp agent messages."""
import derp.util
import derp.camera


class Writer:
    """The disk writer class that records all derp agent messages."""

    def __init__(self, config):
        """Using a dict config connects to all possible subscriber sources"""
        self.config = config
        self.__context, self.__subscriber = derp.util.subscriber(
            [
                "/tmp/derp_camera",
                "/tmp/derp_imu",
                "/tmp/derp_brain",
                "/tmp/derp_joystick",
                "/tmp/derp_keyboard",
            ]
        )
        self.files = {}

    def __del__(self):
        """Close the only active object; the subscriber"""
        self.__subscriber.close()
        self.__context.term()

    def create_rcording(self):
        folder = derp.util.create_record_folder()
        self.files = {
            name: open("%s/%s.bin" % (folder, name), "w+b") for name in derp.util.TOPICS
        }
        with open(str(self.folder / "car.yaml"), "w") as car_fd:
            yaml.dump(self.car_config, car_fd)
        with open(str(self.folder / "brain.yaml"), "w") as brain_fd:
            yaml.dump(self.brain_config, brain_fd)

    def run(self):
        """
        Read the next available topic. If we're not in a recording state and a state message
        arrives with a command to record then create a new folder, and write all messages
        to this folder until another state message tells us to stop.
        """
        topic_bytes, message_bytes = self.__subscriber.recv_multipart()
        topic = topic_bytes.decode()
        message = derp.util.TOPICS[topic].from_bytes(message_bytes).as_builder()

        # Create folder or delete folder
        if topic == "state":
            if message.record and not self.files:
                self.create_recording()
            elif not message.record and self.files:
                for name in self.files:
                    self.files[name].close()
                self.files = {}

        if self.files:
            message.timestampWritten = derp.util.get_timestamp()
            message.write(self.files[topic])


def run(config):
    """A forever while loop to write"""
    writer = Writer(config)
    while True:
        writer.run()
