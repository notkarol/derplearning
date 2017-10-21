import os
import torch.utils.data as data
import numpy as np
import sys
import derputil
sys.path.append('..')
from roadgen3d import Roadgen

class LineFetcher(data.Dataset):

    def __init__(self, root, config, transform=None, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.paths = []

        pipe = Roadgen(config)        
        batch_meta = np.load('%s/batch_meta.npy' % root)
        for i, start in enumerate(batch_meta[:-1]):
            val = pipe.label_norm(np.load('%s/y_%03i.npy' % (root, i)))
            for j, line in enumerate(val):
                self.data.append(line)
                self.paths.append("%s/%09i.png" % (self.root, start + j))

    def __getitem__(self, index):
        img = derputil.load_image(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.data[index]

    
    def __len__(self):
        return len(self.data)
