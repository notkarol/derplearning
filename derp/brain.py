"""
The root class of any object that manipulate's the car state based on some heuristic.
"""
import numpy as np
import torch
import derp.util


class Brain:
    """
    The root class of any object that manipulate's the car state based on some heuristic.
    """

    def __init__(self, config):
        """Preset some common constructor parameters"""
        self.config = config['brain']
        self.car_config = config
        self.messages = [derp.util.TOPICS[topic].new_message() for topic in derp.util.TOPICS]
        self.speed = 0
        self.steer = 0
        self.__sub_context, self.__subscriber = derp.util.subscriber(['/tmp/derp_camera',
                                                                      '/tmp/derp_input',
                                                                      '/tmp/derp_imu'])
        self.__pub_context, self.__publisher = derp.util.subscriber('/tmp/derp_brain')

    def __del__(self):
        self.__subscriber.close()
        self.__sub_context.term() 
        self.__publisher.close()
        self.__pub_context.term() 

    def __repr__(self):
        """Unique instances should not really exist so just use the class name"""
        return self.__class__.__name__.lower()

    def __str__(self):
        """Just use the representation"""
        return repr(self)

    def predict(self):
        return True
    
    def run(self):
        """By default if a child does not override this, do nothing to update the state."""
        topic_bytes, message_bytes = self.__subscriber.recv_multipart()
        recv_timestamp = derp.util.get_timestamp()
        topic = topic_bytes.decode()
        self.messages[topic] = derp.util.TOPICS[topic].from_bytes(message_bytes).as_builder()
        if topic == 'camera' and self.predict():
            msg = derp.util.TOPICS['control'].new_messsage(
                timeCreated=recv_timestamp,
                timePublished=derp.util.get_timestamp(),
                speed=self.speed,
                steer=self.steer,
                manual=False,
            )
            self.__publisher.send_multipart([b'control', msg.to_bytes()])

class Clone(Brain):
    def __init__(self, config):
        super(Clone, self).__init__(config)
        self.camera_config = self.car_config['camera']

        # Show the user what we're working with
        derp.util.print_image_config("Source", self.camera_config)
        derp.util.print_image_config("Target", self.config["thumb"])
        for key in sorted(self.camera_config):
            print("Camera %s: %s" % (key, self.camera_config[key]))
        for key in sorted(self.config["thumb"]):
            print("Target %s: %s" % (key, self.config["thumb"][key]))

        # Prepare camera inputs
        self.bbox = derp.util.get_patch_bbox(self.config["thumb"], self.camera_config)
        self.size = (config["thumb"]["width"], config["thumb"]["height"])

        # Prepare model
        self.model_dir = derp.util.get_brain_config_path(self.config["name"])
        self.model_path = derp.util.find_matching_file(self.model_dir, "clone.pt$")
        if self.model_path is not None and self.model_path.exists():
            self.model = torch.load(str(self.model_path))
            self.model.eval()
        else:
            self.model = None
            print("Clone: Unable to find model path [%s]" % self.model_path)

        # Useful variables for params
        self.prev_steer = 0
        self.prev_speed = 0

        # Data saving
        self.frame_counter = 0

    def prepare_thumb(self, frame):
        if frame is not None:
            patch = derp.util.crop(frame, self.bbox)
            thumb = derp.util.resize(patch, self.size)
        else:
            dim = [self.config["thumb"]["height"], self.config["thumb"]["width"]]
            if self.config["thumb"]["depth"] > 1:
                dim += [self.config["thumb"]["depth"]]
            thumb = np.zeros(dim, dtype=np.float32)
        return thumb

    def predict(self):
        status = derp.util.extractList(self.config["status"], self.state)
        frame = self.state[self.config["thumb"]["component"]]
        self.state["thumb"] = self.prepare_thumb(frame)
        status_batch = derp.util.prepareVectorBatch(status)
        thumb_batch = derp.util.prepareImageBatch(self.state["thumb"])
        status_batch = derp.util.prepareVectorBatch(status)
        if self.model:
            prediction_batch = self.model(thumb_batch, status_batch)
            prediction = derp.util.unbatch(prediction_batch)
            derp.util.unscale(self.config["predict"], prediction)
        else:
            prediction = np.zeros(len(self.config["predict"]), dtype=np.float32)
        self.state["prediction"] = prediction


class CloneAdaSpeed(Clone):
    def __init__(self, config, car_config, state):
        super(CloneAdaSpeed, self).__init__(config, car_config, state)

    def run(self):
        self.predict()
        if self.state["auto"]:
            return

        # Future steering angle magnitude dictates speed
        if self.config["use_min_for_speed"]:
            # predict is assumed to be a list of present and future stearing values
            future_steer = float(min(abs(self.state["prediction"])))
        else:
            future_steer = float(abs(self.state["prediction"][1]))
        multiplier = 1 + self.config["scale"] * (1 - future_steer) ** self.config["power"]

        self.state["speed"] = self.state["offset_speed"] * multiplier
        self.state["steer"] = float(self.state["predictions"][0])


class CloneFixSpeed(Clone):
    def __init__(self, config, car_config, state):
        super(CloneFixSpeed, self).__init__(config, car_config, state)

    def run(self):
        self.predict()
        if not self.state["auto"]:
            return
        self.state["speed"] = self.state["offset_speed"]
        self.state["steer"] = float(self.state["prediction"][0])
