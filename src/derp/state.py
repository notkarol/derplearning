# A class that carries the state of the car through time

from collections.abc import Mapping
from derp.component import Component
import derp.util

class State(Component, Mapping):

    def __init__(self, config, full_config):
        super(State, self).__init__(config)
        self.__state = {}
        self.__exit = False

        # Prepare default state variables
        self['timestamp'] = derp.util.get_current_timestamp()
        self['record'] = False
        self['speed_auto'] = False
        self['steer_auto'] = False
        for var in ['speed_offset', 'steer_offset', 'speed', 'steer']:
            self[var] = config[var] if var in config else 0

    def __getitem__(self, key):
        return self.__state[key]


    def __iter__(self):
        return iter(self.__state)


    def __len__(self):
        return len(self.__state)


    def __setitem__(self, key, item):
        # When we update the state, keep track of every variable we add to the state
        # so when it's saved it's also written to disk. Skip folder as it's long.
        if key not in self.__state:
            self.csv_headers.append(key)
            print("State: Adding [%s]")

        # Update folder if we set record
        if key == 'record' and not self[key]:
            self.__folder = derp.util.get_record_folder()
            
        # Otherwise treat state exactly as a dictionary
        self.__state[key] = item
        return item


    def close(self):
        """
        Mark this run as being done.
        """
        self.__exit = True


    def done(self):
        """
        Returns whether this component is ready to close
        """
        return self.__exit


    def update_multipart(basename, subnames, values):
        """ 
        Sometimes we want to update multiple similarly named variables
        """
        for subname, value in zip(subname, values):
            name = '%s_%s' % (basename, subname)
            self[name] = value
