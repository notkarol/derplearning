"""The disk writer class that records all derp agent messages."""
from derp.part import Part
import derp.util

class Writer(Part):
    """The disk writer class that records all derp agent messages."""

    def __init__(self, config):
        """Using a dict config connects to all possible subscriber sources"""
        super(Writer, self).__init__(config, "writer", ["brain", "camera", "imu", "joystick"])
        self._files = {
            topic: derp.util.topic_file_writer(self._global_config['recording_path'], topic)
            for topic in derp.util.TOPICS
        }

    def __del__(self):
        super(Writer, self).__del__()
        for name in self._files:
            self._files[name].close()

    def run(self):
        """
        Read the next available topic. If we're not in a recording state and a state message
        arrives with a command to record then create a new folder, and write all messages
        to this folder until another state message tells us to stop.
        """
        topic = self.subscribe()
        self._messages[topic].writeNS = derp.util.get_timestamp()
        self._messages[topic].write(self._files[topic])
        return topic != "controller" or not self._messages[topic].exit
