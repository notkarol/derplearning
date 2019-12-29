"""
Fetcher is an image-loader for use with training.
"""
import csv
import numpy as np
import PIL.Image
import torch.utils.data
import derp.util

class Fetcher(torch.utils.data.Dataset):
    """
    Fetcher is an image-loader for use with training.
    """

    def __init__(self, root, transforms, predict_config):
        """
        Our data fetcher is responsible for handling data input to the model training.
        It loads each image in the dataset along with the states for that image.
        Since we use a feed dict, each state variable is stored as a mapping from its
        string name to its value. It is then the responsibility of the data loader or
        training script to properly convert to an array that can be optimized against.
        """

        # Store constructor arguments
        self.root = root.expanduser()
        print(self.root)
        self.transforms = transforms
        self.predict_config = predict_config

        # Pepare variables to store each item
        self.paths = []
        self.status = []
        self.predict = []

        # Read in states and paths
        # Each video has a certain fixed number of state variables which we will encode as a dict
        for recording_name in sorted(self.root.glob("recording-*")):
            recording_path = self.root / recording_name
            status_path = recording_path / "status.csv"
            assert status_path.exists()
            predict_path = recording_path / "predict.csv"
            assert predict_path.exists()

            with open(str(status_path)) as status_fd, open(str(predict_path)) as predict_fd:
                sp_reader, pp_reader = csv.reader(status_fd), csv.reader(predict_fd)
                for status_row, predict_row in zip(sp_reader, pp_reader):
                    assert status_row[0] == predict_row[0]
                    image_path = recording_path / status_row[0]
                    status = np.array([float(x) for x in status_row[1:]], dtype=np.float32)
                    predict = np.array([float(x) for x in predict_row[1:]], dtype=np.float32)
                    if not status:
                        status = np.zeros(1, dtype=np.float32)
                    self.paths.append(image_path)
                    self.status.append(status)
                    self.predict.append(predict)

    def __getitem__(self, index):
        """ Return the specified index. Apply transforms as specified """

        thumb = PIL.Image.fromarray(derp.util.load_image(self.paths[index]))
        status = self.status[index]
        predict = self.predict[index]
        if self.transforms is not None:
            thumb = self.transforms(thumb)
        return thumb, status, predict

    def __len__(self):
        """ Return the number of items our fetcher is responsible for """
        return len(self.paths)
