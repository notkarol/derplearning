"""
The root class of any object that manipulate's the car state based on some heuristic.
"""

class Controller:
    """
    The root class of any object that manipulate's the car state based on some heuristic.
    """

    def __init__(self, config, car_config, state):
        """
        Preset some common constructor parameters.
        """
        self.config = config
        self.car_config = car_config
        self.state = state
        self.ready = True

    def __repr__(self):
        """
        Unique instances should not really exist so just use the class name.
        """
        return self.__class__.__name__.lower()

    def __str__(self):
        """
        Just use the representation.
        """
        return repr(self)

    def plan(self):
        """
        By default if a child does not override this, do nothing to update the state.
        """
        return True
