# A class that carries the state of the car through time

from time import time
from collections.abc import Mapping

class State(Mapping):

    def __init__(self):
        self.state = {'speed': [time(), 0],
                      'steer': [time(), 0],
                      'auto_speed': [time(), False],
                      'auto_steer': [time(), False],
                      'record': [time(), False],
                      'folder': [time(), False],
                      'alert': [time(), False],
                      'steer_offset': [time(), 0]}

        
    def __getitem__(self, key):
        return self.state[key]


    def __setitem__(self, key, item):
        self.state[key] = item
        return item


    def __iter__(self):
        return iter(self.state)


    def __len__(self):
        return len(self.state)


    def __str__(self):
        out = []
        for key in self.state:
            if type(self.state[key]) is str:
                out.append("%s: %s" % (key, self.state[key]))
            elif len(self.state[key]) > 1:
                out.append("%s: |%i|" % (key, len(self.state[key])))
            elif type(self.state[key]) is float:
                out.append("%s: %.3f" % (key, self.state[key]))
            elif type(self.state[key]) is int:
                out.append("%s: %i" % (key, self.state[key]))
            else:
                out.append("%s: %s" % (key, str(self.state[key])))
        return ", ".join(out)
