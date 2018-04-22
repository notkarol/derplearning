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

    def __repr__(self):
        return "%s_%s" % (self.__class__.__name__.lower(), self.config['name'])

    def __str__(self):
        return repr(self)

    def sense(self):
        return True

    def act(self):
        return True
