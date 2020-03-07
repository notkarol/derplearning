"""
The root class of any object that manipulate's the car state based on some heuristic.
"""
import numpy as np
import torch
from derp.part import Part
import derp.util


class Brain(Part):
    """
    The root class of any object that manipulate's the car state based on some heuristic.
    """

    def __init__(self, config):
        """Preset some common constructor parameters"""
        super(Brain, self).__init__(config, "brain", ["camera", "joystick", "imu", "keyboard"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.speed = 0
        self.steer = 0

    def predict(self):
        return True

    def run(self):
        """ Publish an action at every camera timestamp """
        topic = self.subscribe()
        if topic == "camera" and self.predict():
            self.publish("action", isManual=False, speed=float(self.speed), steer=float(self.steer))
        if topic == "controller" and self._messages[topic].exit:
            return False
        return True

    def batch_vector(self, vector):
        numpy_batch = np.reshape(vector, [1, len(vector)])
        torch_batch = torch.from_numpy(numpy_batch).float().to(self.device)
        return torch_batch

    def batch_tensor(self, tensor):
        numpy_batch = np.reshape(tensor, [1] * (4 - len(tensor.shape)) + list(tensor.shape))
        torch_batch = torch.from_numpy(numpy_batch.transpose((0, 3, 1, 2))).float().to(self.device)
        return torch_batch

    def unbatch(self, batch):
        if torch.cuda.is_available():
            out = batch.data.cpu().numpy()
        else:
            out = batch.data.numpy()
        return out


class Clone(Brain):
    def __init__(self, config):
        super(Clone, self).__init__(config)
        model_path = derp.util.MODEL_ROOT / self._config["name"] / "model.pt"
        self.model = torch.load(model_path).to(self.device) if model_path.exists() else None
        self.bbox = derp.util.get_patch_bbox(self._config["thumb"], self._global_config["camera"])
        self.size = (self._config["thumb"]["width"], self._config["thumb"]["height"])

    def predict(self):
        if self.model is None:
            return False
        frame = derp.util.decode_jpg(self._messages["camera"].jpg)
        patch = derp.util.crop(frame, self.bbox)
        thumb = derp.util.resize(patch, self.size) / 255
        status_batch = self.batch_vector([])
        thumb_batch = self.batch_tensor(thumb)

        prediction_batch = self.model(thumb_batch, status_batch)
        predictions = self.unbatch(prediction_batch)[0]

        self.speed = self._messages["controller"].speedOffset
        for prediction, config in zip(predictions, self._config["predict"]):
            if config["name"] == "steer":
                self.steer = float(prediction)
            elif config["name"] == "speed":
                self.speed = float(prediction)
            elif config["name"] == "future_steer":
                self.speed *= 1 + (1 - abs(float(prediction)))
        return True
