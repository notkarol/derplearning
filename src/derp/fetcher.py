import os
import torch.utils.data as data
import derp.util

class Fetcher(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.paths = []
        state_path = os.path.join(self.root, 'states.csv')
        if not os.path.exists(state_path):
            raise(RuntimeError("Unable to find state path [%s]" % state_path))
        with open(state_path) as f:
            headers = f.readline()[:-1].split(',')
            for line in f.readlines():
                row = line[:-1].split(',')
                self.paths.append(os.path.join(self.root, "%s.png" % row[0]))
                self.data.append({k : float(v) for k, v in zip(headers[1:], row[1:])})


    def __getitem__(self, index):
        img = derp.util.load_image(self.paths[index])
        if self.transform is not None:
            img = self.transform(img)

        #import cv2
        #import numpy as np
        #arr = np.array(np.transpose(img.numpy(), (1, 2, 0)) * 255, dtype=np.uint8)
        #arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('%06i_%03i.png' % (index, int(np.random.randint(1000))), arr)

        return img, self.data[index]

    
    def __len__(self):
        return len(self.data)
