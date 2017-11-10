import sys

class Component:

    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.connected = False
        self.out_buffer = []
        self.out_csv_fp = None
        self.folder = None

    def __del__(self):
        print("UNINITIALIZED __del__ %s" % self.__class__.__name__, file=sys.stderr)

        
    def __repr__(self):
        return "(%s, %s)" % (self.name, self.connected)

    
    def __str__(self):
        return repr(self)

    
    # Responsible for updating settings or acting upon the world
    def act(self, state):
        print("UNINITIALIZED act %s" % self.__class__.__name__, file=sys.stderr)
        return False

    
    # Responsible for finding and connecting to the appropriate sensor[s]
    def discover(self):
        print("UNINITIALIZED discover %s" % self.__class__.__name__, file=sys.stderr)
        return False

    
    def scribe(self, folder):
        """ Update the recorders to use the specified folder """
        print("UNINITIALIZED record %s" % self.__class__.__name__, file=sys.stderr)
        return False

    
    def sense(self, state):
        """ Read in sensor data """
        print("UNINITIALIZED sense %s" % self.__class__.__name__, file=sys.stderr)
        return False

    
    def write(self, state):
        """ Write sensor data """
        print("UNINITIALIZED write %s" % self.__class__.__name__, file=sys.stderr)
        return False
