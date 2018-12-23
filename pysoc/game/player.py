
class Player():
    """A Player consists of a name, a Strategy, and possibly other information."""
    def __init__(self, name = None, strategy = None):
        self.name = name 
        self.strategy = strategy 
    def __repr__(self):
        name = '<Player>' if (self.name is None) else self.name 
        if self.strategy is None:
            return name 
        return '{}: {!r}'.format(name, self.strategy)