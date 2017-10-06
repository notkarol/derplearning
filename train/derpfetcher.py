import torch.utils.data as data
from PIL import Image
import os
import derputil

class DerpFetcher(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.paths = []
        state_path = os.path.join(folder, 'states.csv')
        if not os.path.exists(state_path):
            raise(RuntimeError("Unable to find state path [%s]" % state_path))
        with open(state_path) as f:
            headers = f.readline().split(',')
            for line in f.readlines():
                row = line.split(',')
                self.paths.append(os.path.join(folder, "%s.png" % row[0]))
                self.data.append({k : float(v) for k, v in zip(headers[1:], row[1:])})


    def __getitem__(self, index):
        img = self.loadImage(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.data[index]

    
    def __len__(self):
        return len(self.data)
