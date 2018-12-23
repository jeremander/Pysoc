"""Binary Symmetric Games where each player has 2 choices, Cooperate or Defect:
    P (Penalty) for (Defect, Defect)
    T (Temptation) for (Defect, Cooperate)
    S (Sucker) for (Cooperate, Defect)
    R (Reward) for (Cooperate, Cooperate)"""

import numpy as np 

from pysoc.game.game import TwoPlayerBinaryGame
from pysoc.game.strategy import StrategyParams, Strategy, PureSymmetricStrategy, MixedSymmetricStrategy
from pysoc.misc.cache import memoize

#########
# GAMES #
#########

class TwoPlayerBinarySymmetricGame(TwoPlayerBinaryGame):
    """Two-player game where each player has two choices,
    and their positions are symmetric, meaning the payoff matrix must satisfy P0(i, j) = P1(j, i)."""
    def __init__(self, mat, name = 'Untitled', action_names = ('D', 'C')):
        """Payoff matrix is two-dimensional, indicating player 0's payoffs.
        Player 0's payoffs obey the symmetry relation.
        action_names is a single sequence of names of both players' actions."""
        mat = np.array(mat)
        if (mat.shape != (2, 2)):
            raise ValueError("A single payoff matrix to player 0 is required.")
        mat = np.array([mat, mat.T])
        action_names = None if (action_names is None) else (action_names, action_names)
        super().__init__(mat, name, action_names)
    @classmethod
    def from_PTSR(cls, P, T, S, R, name = 'Untitled', action_names = ('D', 'C')):
        """Initialize from 4 values:
        P (Penalty) for (Defect, Defect)
        T (Temptation) for (Defect, Cooperate)
        S (Sucker) for (Cooperate, Defect)
        R (Reward) for (Cooperate, Cooperate)"""
        return cls([[P, T], [S, R]], name, action_names)

@memoize
class PrisonersDilemma(TwoPlayerBinarySymmetricGame):
    """The game of Prisoner's Dilemma (S < P < R < T).
    User may set the exact payoffs."""
    def __init__(self, P = 1, T = 3, S = 0, R = 2, name = "Prisoner's Dilemma", action_names = ('D', 'C')):
        if (not (S < P < R < T)):
            raise ValueError("Prisoner's Dilemma must have S < P < R < T.")
        super().__init__([[P, T], [S, R]], name, action_names)

@memoize
class Chicken(TwoPlayerBinarySymmetricGame):
    """The game of Chicken (P < S < R < T).
    User may set the exact payoffs."""
    def __init__(self, P = 0, T = 3, S = 1, R = 2, name = "Chicken", action_names = ('D', 'C')):
        if (not (P < S < R < T)):
            raise ValueError("Chicken must have P < S < R < T.")
        super().__init__([[P, T], [S, R]], name, action_names)

@memoize
class StagHunt(TwoPlayerBinarySymmetricGame):
    """The game of Stag Hunt (S < P < T < R).
    User may set the exact payoffs."""
    def __init__(self, P = 1, T = 2, S = 0, R = 3, name: str = "Stag Hunt", action_names = ('D', 'C')):
        if (not (S < P < T < R)):
            raise ValueError("Stag Hunt must have P < S < R < T.")
        super().__init__([[P, T], [S, R]], name, action_names)

@memoize
class CoordinationGame(TwoPlayerBinarySymmetricGame):
    """A coordination game (S = T < P = R)."""
    def __init__(self, P = 1, T = 0, S = 0, R = 1, name = "Coordination Game", action_names = ('A', 'B')):
        if (not (S == T < P == R)):
            raise ValueError("Coordination Game must have S == T < P == R.")
        super().__init__([[P, T], [S, R]], name, action_names)

##############
# STRATEGIES #
##############

DEFECT = 0
COOPERATE = 1

class PTSRStrategy(Strategy):
    """Strategy for playing a PTSR game."""
    VALID_GAME = TwoPlayerBinarySymmetricGame

class Defector(PureSymmetricStrategy, PTSRStrategy):
    """Defects all the time."""
    def __init__(self):
        super().__init__(DEFECT)

class Cooperator(PureSymmetricStrategy, PTSRStrategy):
    """Cooperates all the time."""
    def __init__(self):
        super().__init__(COOPERATE)

class Random(MixedSymmetricStrategy, PTSRStrategy):
    """Cooperates randomly with probability p, otherwise defects."""
    def __init__(self, p):
        super().__init__([1 - p, p])
        self.p = p  # for __repr__

class TitForTat(PTSRStrategy):
    """Initially cooperates, then does whatever the opponent did last."""
    PARAMS = StrategyParams(memory_depth = 1, match_data = set())
    def play_turn(self):
        """Note: don't need to be aware of the turn in general, just need to default to cooperation on the first turn."""
        try:
            return self.match().players[1 - self.index].action_history[-1]
        except IndexError:
            return COOPERATE

