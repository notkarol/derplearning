import os
import csv
import derp.util

class Component:

    def __init__(self):
        raise ValueError("Please do not use default constructor; supply a config")


    def __init__(self, config, full_config):
        # Common variables
        self.__config = config
        self.__ready = False

        # Output csv variables
        self.__folder = None
        self.__csv_fd = None
        self.__csv_writer = None
        self.__csv_buffer = []
        self.__csv_header = []


    def __del__(self):
        if self.__csv_fd is not None:
            self.__csv_fd.close()


    def __repr__(self):
        return "%s_%s_%s" % (self.__class__.__name__, self.__config['name'], self.__ready)


    def __str__(self):
        return repr(self)


    def __is_recording(self, state):
        return state['record']


    def __is_recording_initialized(self, state):
        return state['folder'] == self.__folder


    def ready(self):
        """
        Returns whether this component is ready to be used
        """
        return self.__ready


    def sense(self, state):
        return True
    

    def plan(self, state):
        return True


    def act(self, state):
        return True


    def record(self, state):
        """
        Creates the output csv file
        If it returns true, that means that it is good to write outputs
        """

        # Skip if aren't asked to record or we have nothing to record
        if not self.__is_recording(state):
            return False

        # Create a new output csv writer since the folder name changed
        if len(self.__csv_header):
            if not self.__is_recording_initialized(state) and
                self.__folder = state['folder']

                # Close existing csv file descriptor if it exists
                if self.__csv_fd is not None:
                    self.__csv_fd.close()

                # Create a new folder if we don't have one yet
                derp.util.mkdir(self.__folder)

                # Create output csv
                filename = "%s.csv" % (str(self).lower())
                csv_path = os.path.join(self.__folder, filename)
                self.__csv_fd = open(csv_path, 'w')
                self.__csv_writer = csv.writer(self.__csv_fd, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL)
                self.__csv_writer.write(out.csv_header)

            # Write out buffer and flush it
            for row in self.__csv_buffer:
                self.__csv_writer.write(row)
            self.__csv_fd.flush()

        # Clear csv buffer in any case to prevent memory leaks
        del self.__csv_buffer[:]

        return True
