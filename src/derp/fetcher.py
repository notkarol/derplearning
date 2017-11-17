import csv
import numpy as np
from os.path import expanduser, exists, join, basename, splitext
import torch.utils.data
import derp.util

class Fetcher(torch.utils.data.Dataset):

    def __init__(self, root, transform_x=None, transform_xy=None):
        """ 
        Our data fetcher is responsible for handling data input to the model training.
        It loads each image in the dataset along with the states for that image.
        Since we use a feed dict, each state variable is stored as a mapping from its
        string name to its value. It is then the responsibility of the data loader or
        training script to properly convert to an array that can be optimized against.
        """

        # Store constructor arguments
        self.root = expanduser(root)
        self.transform_x = transform_x
        self.transform_xy = transform_xy

        # Pepare variables to store each item
        self.paths = []
        self.states = []

        # Make sure we can find the path
        state_path = join(self.root, 'states.csv')
        if not exists(state_path):
            raise(RuntimeError("Fetcher: Unable to find state path [%s]" % state_path))

        # Read in states and paths
        # Each video has a certain fixed number of state variables which we will encode as a dict
        with open(state_path) as f:
            reader = csv.reader(f)
            for row in reader:
                path = join(self.root, row[0])
                state = np.array([float(x) for x in row[1:]], dtype=np.float32)
                self.paths.append(path)
                self.states.append(state)


    def __getitem__(self, index):
        """ Return the specified index. Apply transforms as specified """
        
        # Prepare 
        x = derp.util.load_image(self.paths[index])
        y = self.states[index].copy()
        
        # Transform x
        if self.transform_x is not None:
            x = self.transform_x(x)

        # Transform x and y
        if self.transform_xy is not None:
            x, y = self.transform_xy((x, y))
            
        return x, y

    
    def __len__(self):
        """ Return the number of items our fetcher is responsible for """
        return len(self.paths)
