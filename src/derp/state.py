# A class that carries the state of the car through time

from collections.abc import Mapping
from shutil import copyfile
import numpy as np
import os

class State(Mapping):

    def __init__(self, speed_offset=0.0, steer_offset=0.0):
        self.state = {'folder': None,
                      'record': False,
                      'speed': 0,
                      'steer': 0,
                      'auto_speed': False,
                      'auto_steer': False,
                      'speed_offset': speed_offset,
                      'steer_offset': steer_offset,
                      'exit': False}
        self.folder = None
        self.out_csv_fp = None

        
    def __del__(self):
        if self.out_csv_fp is not None:
            self.out_csv_fp.close()
            
        
    def __getitem__(self, key):
        return self.state[key]


    def __setitem__(self, key, item):
        self.state[key] = item
        return item


    def __iter__(self):
        return iter(self.state)


    def __len__(self):
        return len(self.state)

    
    def __repr__(self):
        if 'timestamp' not in self.state:
            return 'uninitialized'
        out = str(self.state['timestamp'])
        for key in sorted(self.state):
            if key == 'timestamp' or key == 'folder':
                continue
            val = self.state[key]
            out += ','
            if type(val) is int:
                out += str(val)
            elif type(val) is float:
                s = str(val)
                if len(s) > 9:
                    s = "%.6f" % val
                out += s
            elif type(val) is bool:
                out += str(int(val))
            else:
                out += 'nan'
        return out
        

    def __str__(self):
        out = []
        for key in self.state:
            val = self.state[key]
            if type(val) is str or type(val) is bool:
                out.append("(%s: %s)" % (key, val))
            elif type(val) is float:
                out.append("(%s: %.3f)" % (key, val))
            elif type(val) is int:
                out.append("(%s, %i)" % (key, val))
            else:
                out.append("(%s)" % key)
        return ", ".join(out)

    
    def scribe(self, hw_config_path):

        # Do not do anything if recording is off
        if not self.state['record']:
            return False

        # If we already have this folder, write out the state
        if self.folder != self.state['folder']:

            # Create the folder
            self.folder = self.state['folder']
            os.mkdir(self.state['folder'])
            print('Created:', self.state['folder'])

            # Make a copy of our config
            copyfile(hw_config_path, os.path.join(self.folder, 'config.yaml'))

            # Prepare output csv
            out_csv_path = os.path.join(self.folder, "state.csv")
            self.out_csv_fp = open(out_csv_path, 'w')

            # Write out headers
            self.out_csv_fp.write("timestamp")
            for key in sorted(self.state):
                if key == 'timestamp':
                    continue
                self.out_csv_fp.write(',' + key)
            self.out_csv_fp.write("\n")
            self.out_csv_fp.flush()

        # Write out the first row of data
        self.write()
        return True
        
    def write(self):
        self.out_csv_fp.write(repr(self) + "\n")
        self.out_csv_fp.flush()
        return True
