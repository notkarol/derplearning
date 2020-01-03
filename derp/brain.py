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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.messages = {topic: derp.util.TOPICS[topic].new_message() for topic in derp.util.TOPICS}
        self.speed = 0
        self.steer = 0
        self.__sub_context, self.__subscriber = derp.util.subscriber(['/tmp/derp_camera',
                                                                      '/tmp/derp_joystick',
                                                                      '/tmp/derp_imu'])
        self.__pub_context, self.__publisher = derp.util.publisher('/tmp/derp_brain')

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
            msg = derp.util.TOPICS['action'].new_message(
                timeCreated=recv_timestamp,
                timePublished=derp.util.get_timestamp(),
                isManual=False,
                speed=float(self.speed),
                steer=float(self.steer),
            )
            self.__publisher.send_multipart([b'action', msg.to_bytes()])
            
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
        model_path = derp.util.ROOT / 'scratch' / self.config['name'] / 'model.pt'
        self.model = torch.load(model_path).to(self.device) if model_path.exists() else None
        self.camera_config = self.car_config['camera']
        self.bbox = derp.util.get_patch_bbox(self.config['thumb'], self.camera_config)
        self.size = (self.config['thumb']['width'], self.config['thumb']['height'])
        
    def predict(self):
        if self.model is None:
            return False
        frame = derp.util.decode_jpg(self.messages['camera'].jpg)
        patch = derp.util.crop(frame, self.bbox)
        thumb = derp.util.resize(patch, self.size) / 255
        status_batch = self.batch_vector([])
        thumb_batch = self.batch_tensor(thumb)

        prediction_batch = self.model(thumb_batch, status_batch)
        predictions = self.unbatch(prediction_batch)[0]

        self.speed = self.messages['controller'].speedOffset
        for prediction, config in zip(predictions, self.config['predict']):
            if config['name'] == 'steer':
                self.steer = float(prediction)
            elif config['name'] == 'speed':
                self.speed = float(prediction)
            elif config['name'] == 'future_steer':
                self.speed *= 1 + (1 - abs(float(prediction)))
        return True

def loop(config):
    brain_class = eval(config['brain']['class'])
    brain = brain_class(config)
    while True:
        brain.run()
    
