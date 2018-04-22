class Component:
    def __init__(self, config, state):
        self.config = config
        self.state = state
        self.ready = False

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__.lower(), self.config['name'])

    def __str__(self):
        return repr(self)

    def sense(self):
        return True

    def act(self):
        return True
