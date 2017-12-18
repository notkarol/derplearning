# A class that carries the state of the car through time

from collections.abc import Mapping
import os
import time
from derp.component import Component
import derp.util

class State(Component, Mapping):

    def __init__(self, config, full_config):
        super(State, self).__init__(config, full_config)
        self.state = {}
        self.exit = False
        self.folder = None
        self.config['name'] = 'state'

        # Prepare default state variables
        self['timestamp'] = time.time()
        self['record'] = False
        self['folder'] = None
        self['auto_speed'] = False
        self['auto_steer'] = False
        self['speed'] = 0
        self['steer'] = 0
        self['offset_speed'] = 0
        self['offset_steer'] = 0


    def __getitem__(self, key):
        return self.state[key]


    def __iter__(self):
        return iter(self.state)


    def __len__(self):
        return len(self.state)

    
    def __repr__(self):
        return self.__class__.__name__.lower()

    
    def __setitem__(self, key, item):
        # When we update the state, keep track of every variable we add to the state
        # so when it's saved it's also written to disk. Skip folder as it's long.
        if key not in self.state:
            if key not in ['record', 'folder']:
                self.csv_header.append(key)
            print("state created: %s" % key)

        # Update folder if we set record
        if key == 'record' and key in self.state:
            if item:
                if not self[key]:
                    self['folder'] = derp.util.create_record_folder()
            else:
                self['folder'] = None
                
            
        # Otherwise treat state exactly as a dictionary
        self.state[key] = item
        return item

    
    def record(self, state=None):
        if self.is_recording(self):
            row = []
            for key in self.csv_header:
                t = type(self[key])
                if t in (int, bool, type(None)):
                    row.append(self[key])
                elif t is float:
                    row.append(("%.6f" % self[key]).rstrip('0'))
                else:
                    row.append('')
            self.csv_buffer.append(row)
        super(State, self).record(self)
            

    def close(self):
        """
        Mark this run as being done.
        """
        self.exit = True


    def done(self):
        """
        Returns whether this component is ready to close
        """
        return self.exit


    def update_multipart(self, basename, subnames, values):
        """ 
        Sometimes we want to update multiple similarly named variables
        """
        for subname, value in zip(subnames, values):
            name = '%s_%s' % (basename, subname)
            self[name] = value
