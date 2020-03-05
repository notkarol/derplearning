"""
A part is a component of the overall derp system that communicates with other parts
"""
from derp.util import TOPICS, MSG_STEM, subscriber, publisher, get_timestamp
import logging


class Part:
    """ The root class for every part, includes a bunch of useful functions and cleanup """

    def __init__(self, config, name, sub_names):
        """ By default every part is its own publisher and subscribes to one/many messages """
        self._name = name
        self._config = config[name]
        self._global_config = config
        self._messages = {topic: TOPICS[topic].new_message() for topic in TOPICS}
        self._sub_context, self._subscriber = subscriber([MSG_STEM + name for name in sub_names])
        self._pub_context, self._publisher = publisher(MSG_STEM + name)
        self._timestamp = 0

    def __del__(self):
        """ Clean up the pub/sub system """
        self._subscriber.close()
        self._sub_context.term()
        self._publisher.close()
        self._pub_context.term()
        print("__DEL__", self.__class__.__name__)

    def __repr__(self):
        return self.__class__.__name__.lower()

    def __str__(self):
        return repr(self)

    def run(self):
        assert True

    def subscribe(self):
        topic_bytes, message_bytes = self._subscriber.recv_multipart()
        self._timestamp = get_timestamp()
        topic = topic_bytes.decode()
        if topic == "":
            return None
        self._messages[topic] = TOPICS[topic].from_bytes(message_bytes).as_builder()
        return topic

    def publish(self, topic, **kwargs):
        message = TOPICS[topic].new_message(
            createNS=self._timestamp, publishNS=get_timestamp(), **kwargs
        )
        self._publisher.send_multipart([str.encode(topic), message.to_bytes()])
