# A class that carries the state of the car through time
from collections.abc import Mapping
import csv
import numpy as np
import os
import yaml
import time
import derp.util

class State(Mapping):

    def __init__(self, car_config, controller_config):
        self.exit = False
        self.folder = None
        self.car_config = car_config
        self.controller_config = controller_config
        self.state = {'record': False}
        self.csv_fd = None
        self.csv_writer = None
        self.csv_header = []
        
        # Prepare default state variables
        self['timestamp'] = time.time()
        self['warn'] = False
        self['error'] = False
        self['auto'] = False
        self['speed'] = 0
        self['steer'] = 0
        self['offset_speed'] = 0.0
        self['offset_steer'] = 0.0
        self['use_offset_speed'] = False
        self['use_offset_steer'] = True
        self['frame_counter'] = 0

    def __getitem__(self, key):
        return self.state[key]

    def __iter__(self):
        return iter(self.state)

    def __len__(self):
        return len(self.state)
    
    def __repr__(self):
        return self.__class__.__name__.lower()

    def __setitem__(self, key, item):

        if key not in self.state:
            if self.state['record']:
                raise KeyError("Cannot create variable [%s] during recording" % key)
            self.csv_header.append(key)
            print("state created: %s" % key)

        # Update folder if we set record
        if key == 'record' and key in self.state and item and not self[key]:
            self.initialize_recording()

        # Otherwise treat state exactly as a dictionary
        self.state[key] = item
        return item

    def initialize_recording(self):
        self.folder = derp.util.create_record_folder()
        self['frame_counter'] = 0
        
        dst_car_config_path = os.path.join(self.folder, 'car.yaml')
        with open(dst_car_config_path, 'w') as f:
            yaml.dump(self.car_config, f)

        dst_controller_config_path = os.path.join(self.folder, 'controller.yaml')
        with open(dst_controller_config_path, 'w') as f:
            yaml.dump(self.controller_config, f)

        # Make a folder for every 2D or larger numpy array so we can store vectors/images
        for key in self.state:
            if self.is_multidimensional(key):
                folder = os.path.join(self.folder, key)
                os.mkdir(folder)
        
        # Create state csv
        csv_path = os.path.join(self.folder, 'state.csv')
        self.csv_fd = open(csv_path, 'w')
        self.csv_writer = csv.writer(self.csv_fd, delimiter=',', quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
        self.csv_writer.writerow(self.csv_header)
        
        
    def is_multidimensional(self, var):
        return type(self[var]) is np.ndarray and len(self[var]) > 1

    def is_recording(self):
        return 'record' in self.state and self.state['record']

    def is_image(self, key):
        return len(self[key].shape) == 3 and self[key].shape[-1] == 3

    def get_image_suffix(self, key):
        return 'jpg' if 'camera' in key else 'png'
    
    def record(self):
        # If we're not recording anymore, do post-processing and stop
        if not self.is_recording():
            if self.folder is not None:
                self.csv_fd.close()
                for key in self.state:
                    if self.is_multidimensional(key) and self.is_image(key):
                        suffix = self.get_image_suffix(key)
                        derp.util.encode_video(self.folder, key, suffix)
                self.folder = None
            return False
        
        # Prepare the csv row to print
        row = []
        for key in self.csv_header:
            t = type(self[key])
            if t in (int, bool, type(None)):
                row.append(self[key])
            elif t is float:
                row.append(("%.6f" % self[key]).rstrip('0'))
            else:
                row.append('')
        self.csv_writer.writerow(row)
        self.csv_fd.flush()

        for key in self.state:
            if not self.is_multidimensional(key):
                continue

            path_stem = os.path.join(self.folder, key, "%06i." % self['frame_counter'])
)
            if self.is_image(key):
                path = path_stem + self.get_image_suffix(key)
                derp.util.save_image(path, self[key])
            else:
                np.save(path_stem, self[key], allow_pickle=False)
    
        self['frame_counter'] += 1
        return True

    def close(self):
        """ Mark this run as being done so that the program knows to exist"""
        self.exit = True

    def done(self):
        """ Returns whether the state is ready to close """
        return self.exit

    def update_multipart(self, basename, subnames, values):
        """ Sometimes we want to update multiple similarly named variables """
        for subname, value in zip(subnames, values):
            name = '%s_%s' % (basename, subname)
            self[name] = value
        

            
