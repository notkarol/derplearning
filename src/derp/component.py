import sys

class Component:

    #initializes the component
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.out_buffer = []
        self.out_csv_fp = None
        self.folder = None

    #deletes the class object
    def __del__(self):
        print("UNINITIALIZED __del__ %s" % self.__class__.__name__, file=sys.stderr)

    
    #returns an unambiguous representation of the object
    def __repr__(self):
        return "(%s, %s)" % (self.config['name'], self.connected)

    
    #returns a human readable representation of the object
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
