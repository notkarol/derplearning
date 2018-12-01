"""
A class that carries the state of the car through time.
"""
from collections.abc import Mapping
import csv
import time
import yaml

import numpy as np

import derp.util

class State(Mapping):
    """
    A class that carries the state of the car through time.
    """

    def __init__(self, car_config, brain_config, debug=False):
        """
        Create the dict that is this is this class and pre-set some defaults.
        """
        self.exit = False
        self.folder = None
        self.car_config = car_config
        self.brain_config = brain_config
        self.state = {'record': False}
        self.csv_fd = None
        self.csv_writer = None
        self.csv_header = []
        self.previous_timestamp = 0

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
        self['debug'] = debug

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
        with open(str(self.folder / 'car.yaml'), 'w') as car_fd:
            yaml.dump(self.car_config, car_fd)
        with open(str(self.folder / 'brain.yaml'), 'w') as brain_fd:
            yaml.dump(self.brain_config, brain_fd)

        # Make a folder for every 2D or larger numpy array so we can store vectors/images
        for key in self.state:
            if self.is_multidimensional(key):
                folder = self.folder / key
                folder.mkdir(parents=True, exist_ok=True)

        # Create state csv
        self.csv_fd = open(str(self.folder / 'state.csv'), 'w')
        self.csv_writer = csv.writer(self.csv_fd, delimiter=',', quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
        self.csv_writer.writerow(self.csv_header)


    def is_multidimensional(self, var):
        return isinstance(self[var], np.ndarray) and len(self[var]) > 1

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
            key_type = type(self[key])
            if key_type in (int, bool, type(None)):
                row.append(self[key])
            elif key_type is float:
                row.append(("%.6f" % self[key]).rstrip('0'))
            else:
                row.append('')
        self.csv_writer.writerow(row)
        self.csv_fd.flush()

        for key in self.state:
            if not self.is_multidimensional(key):
                continue

            path_stem = str(self.folder / key / ("%06i." % self['frame_counter']))
            if self.is_image(key):
                path = path_stem + self.get_image_suffix(key)
                derp.util.save_image(path, self[key])
            else:
                np.save(path_stem + '.npy', self[key], allow_pickle=False)

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

    def print(self):
        """
        Print a short summary of the state for debugging purposes.
        """
        fps = 1 / (self.state['timestamp'] - self.previous_timestamp)
        print("%.3f %.2f %2i %s %s | speed %6.3f + %6.3f %i | steer %6.3f + %6.3f %i" %
              (self.state['timestamp'], self.state['warn'], fps, 
               'R' if self.state['record'] else '_', 'A' if self.state['auto'] else '_',
               self.state['speed'], self.state['offset_speed'], self.state['use_offset_speed'],
               self.state['steer'], self.state['offset_steer'], self.state['use_offset_steer']))
        self.previous_timestamp = self.state['timestamp']
            
