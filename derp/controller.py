class Controller:
    def __init__(self, config, car_config, state):
        self.config = config
        self.car_config = car_config
        self.state = state
        self.ready = True

    def __repr__(self):
        return self.__class__.__name__.lower()

    def __str__(self):
        return repr(self)

    def plan(self):
        return True
