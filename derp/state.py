# A class that carries the state of the car through time
from collections.abc import Mapping
import csv
import os
import yaml
import time
from derp.component import Component
import derp.util

class State(Mapping):

    def __init__(self, config):
        self.exit = False
        self.folder = None
        self.config = config
        self.state = {}
        self.csv_fd = None
        self.csv_writer = None
        self.csv_buffer = []
        self.csv_header = []
        
        # Prepare default state variables
        self['timestamp'] = time.time()
        self['record'] = False
        self['folder'] = self.folder
        self['warn'] = False
        self['error'] = False
        self['auto'] = False
        self['speed'] = 0
        self['steer'] = 0
        self['offset_speed'] = 0.0
        self['offset_steer'] = 0.0
        self['use_offset_speed'] = False
        self['use_offset_steer'] = True


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
                    config_path = os.path.join(self['folder'], 'config.yaml')
                    with open(config_path, 'w') as f:
                        f.write(yaml.dump(self.config, default_flow_style=False))
            else:
                self['folder'] = None
                
            
        # Otherwise treat state exactly as a dictionary
        self.state[key] = item
        return item


    def is_recording(self):
        return 'record' in self.state and self.state['record']


    def is_recording_initialized(self):
        return self.folder is not None  


    def record(self):
        if self.is_recording():
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

        # Skip if aren't asked to record or we have nothing to record
        if not self.is_recording():
            if self.is_recording_initialized():
                self.folder = None
            return False

        # As long as we have a csv header to write out, write out data
        if len(self.csv_header):

            # Create a new output csv writer since the folder name changed
            if not self.is_recording_initialized():
                self.folder = self.state['folder']
                # Close existing csv file descriptor if it exists
                if self.csv_fd is not None:
                    self.csv_fd.close()

                # Create output csv
                filename = "%s.csv" % (str(self).lower())
                csv_path = os.path.join(self.folder, filename)
                self.csv_fd = open(csv_path, 'w')
                self.csv_writer = csv.writer(self.csv_fd, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
                self.csv_writer.writerow(self.csv_header)

            # Write out buffer and flush it
            for row in self.csv_buffer:
                self.csv_writer.writerow(row)
            self.csv_fd.flush()

        # Clear csv buffer in any case to prevent memory leaks
        del self.csv_buffer[:]

        return True
            

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


    def verify(self):
        # Verify that all have been initialized
        for var in self.state:
            if self.state[var] is None and var is not 'folder':
                raise ValueError("Field [%s] is None" % var)
        

            
