"""
A component is the base class for every part of the car.
"""

class Component:
    """
    A component is the base class for every part of the car.
    """

    def __init__(self, config, state):
        """ A parameterized initializer to set default args """
        self.config = config
        self.state = state
        self.ready = False

    def __repr__(self):
        """
        Use the name from the config as this instance's unique name of this class
        """
        return "%s(%s)" % (self.__class__.__name__.lower(), self.config['name'])

    def __str__(self):
        """
        Just use the representation.
        """
        return repr(self)

    def sense(self):
        """
        By default if a child does not override this, do nothing to collect data/state.
        """
        return True

    def act(self):
        """
        By default if a child does not override this, do nothing to actuate anything.
        """
        return True
