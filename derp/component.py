import os
import csv
import derp.util

class Component:

    def __init__(self):
        raise ValueError("Please do not use default constructor; supply a config")

    def __init__(self, config, full_config, state):
        # Common variables
        self.config = config
        self.full_config = full_config
        self.state = state
        self.ready = False

        # Output csv variables
        self.folder = None
        self.csv_fd = None
        self.csv_writer = None
        self.csv_buffer = []
        self.csv_header = []

    def __del__(self):
        if self.csv_fd is not None:
            self.csv_fd.close()

    def __repr__(self):
        return "%s_%s" % (self.__class__.__name__.lower(), self.config['name'])

    def __str__(self):
        return repr(self)

    def is_recording(self):
        return 'record' in self.state and self.state['record']

    def is_recording_initialized(self):
        return self.folder is not None

    def ready(self):
        """
        Returns whether this component is ready to be used
        """
        return self.ready

    def sense(self):
        return True

    def plan(self):
        return True

    def act(self):
        return True

    def flush(self):
        return True

    def record(self):
        """
        Creates the output csv file
        If it returns true, that means that it is good to write outputs
        """

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
