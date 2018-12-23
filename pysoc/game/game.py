import itertools
import numpy as np
import pandas as pd

from pysoc.misc.cache import memoize

class Game():
    """A class to hold game matrices and to assign payoffs to an outcome accordingly."""
    def __init__(self, mat, name = 'Untitled', action_names = None):
        """mat is a sequence of payoff matrices for each player.
        The shape of each payoff matrix dimension has length equal to the number of players,
        and the size of each dimension is equal to the number of choices available to the corresponding player.
        E.g. if it is a 2-player game, where player 0 has 3 choices and player 1 has 2 choices,
        the shape of mat should be [2, 3, 2].
        name is the name of the game.
        action_names is a list of action names for each player."""
        self.mat = np.array(mat)
        self.nplayers = self.mat.shape[0]
        self.num_actions_by_player = self.mat.shape[1:]
        self.name = name
        self.action_names = action_names
        if (action_names is None):
            self.action_names = tuple(tuple(map(str, range(n))) for n in self.num_actions_by_player)
        else:
            if (self.num_actions_by_player != tuple(map(len, action_names))):
                raise ValueError("action_names must match dimensions of payoff matrices.")
            self.action_names = action_names
        self.action_indices_by_name = tuple({action : i for (i, action) in enumerate(names)} for names in self.action_names)
    def payoff(self, actions):
        """Given a sequence of actions by all of the players, returns the tuple of payoffs to each player."""
        if isinstance(actions[0], str):  # convert strings to integers
            actions = [self.action_indices_by_name[i][actions[i]] for i in range(self.nplayers)]
        return tuple(self.mat[tuple([i] + list(actions))] for i in range(self.nplayers))
    def __repr__(self):
        return '{}-player Game "{}" with payoff matrix:\n\n{}'.format(self.nplayers, self.name, self.mat)

class TwoPlayerGame(Game):
    """A two-player game."""
    def __init__(self, mat, name = 'Untitled', action_names = None):
        """Payoff matrix has shape (2, m, n), where m is the number of player 0's actions
        and n is the number of player 1's actions.
        action_names is an optional pair of lists of action names."""
        super().__init__(mat, name, action_names)
        if (self.nplayers != 2):
            raise ValueError("Must input 2 payoff matrices, one for each player.")
    def __repr__(self):
        df = pd.DataFrame(index = self.action_names[0], columns = self.action_names[1])
        [m, n] = self.num_actions_by_player
        for (i, j) in itertools.product(range(m), range(n)):
            df.iloc[i, j] = (self.mat[0, i, j], self.mat[1, i, j])
        return '{}-player Game "{}" with payoff matrix:\n\n{}'.format(self.nplayers, self.name, df)

class TwoPlayerBinaryGame(TwoPlayerGame):
    """Two-player game where each player has two choices."""
    def __init__(self, mat, name = 'Untitled', action_names = None):
        """Payoff matrix must have shape (2, 2, 2)."""
        super().__init__(mat, name, action_names)
        if (self.num_actions_by_player != (2, 2)):
            raise ValueError("Must have only two actions for each player.")

class TwoPlayerZeroSumGame(TwoPlayerGame):
    """A two-player zero-sum game."""
    def __init__(self, mat, name = 'Untitled', action_names = None):
        """Payoff matrix is two-dimensional, indicating only player 0's payoffs.
        Player 1's payoffs are just the negative of player 0's."""
        mat = np.array(mat)
        if (len(mat.shape) != 2):
            raise ValueError("A single payoff matrix to player 0 is required.")
        mat = np.array([mat, -mat])
        super().__init__(mat, name, action_names)
