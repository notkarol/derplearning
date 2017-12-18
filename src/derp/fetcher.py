import csv
import numpy as np
import os
import torch.utils.data
import derp.util

class Fetcher(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        """ 
        Our data fetcher is responsible for handling data input to the model training.
        It loads each image in the dataset along with the states for that image.
        Since we use a feed dict, each state variable is stored as a mapping from its
        string name to its value. It is then the responsibility of the data loader or
        training script to properly convert to an array that can be optimized against.
        """

        # Store constructor arguments
        self.root = os.path.expanduser(root)
        self.transform = transform

        # Pepare variables to store each item
        self.paths = []
        self.status = []
        self.predict = []

        # Read in states and paths
        # Each video has a certain fixed number of state variables which we will encode as a dict
        for recording_name in sorted(os.listdir(self.root)):

            # Skip any non-subpath
            path = os.path.join(self.root, recording_name)
            if not os.path.isdir(path):
                continue
            
            # Make sure we can find the path
            status_path = os.path.join(path, 'status.csv')
            if not os.path.exists(status_path):
                raise(RuntimeError("Fetcher: Unable to find status path"))

            predict_path = os.path.join(path, 'predict.csv')
            if not os.path.exists(predict_path):
                raise(RuntimeError("Fetcher: Unable to find predict path"))

            with open(status_path) as sp, open(predict_path) as pp:
                sp_reader, pp_reader = csv.reader(sp), csv.reader(pp)
                for status_row, predict_row in zip(sp_reader, pp_reader):
                    if status_row[0] != predict_row[0]:
                        raise(ValueError("Fetcher: discrepancy between status and predict"))
                    path = os.path.join(self.root, status_row[0])
                    status = np.array([float(x) for x in status_row[1:]], dtype=np.float32)
                    predict = np.array([float(x) for x in predict_row[1:]], dtype=np.float32)

                    status = status if len(status) else np.zeros(1, dtype=np.float32)
                    self.paths.append(path)
                    self.status.append(status)
                    self.predict.append(predict)


    def __getitem__(self, index):
        """ Return the specified index. Apply transforms as specified """
        
        # Prepare 
        thumb = derp.util.load_image(self.paths[index])
        status = self.status[index]
        predict = self.predict[index]

        # Transform x
        if self.transform is not None:
            thumb = self.transform(thumb)

        return thumb, status, predict

    
    def __len__(self):
        """ Return the number of items our fetcher is responsible for """
        return len(self.paths)
