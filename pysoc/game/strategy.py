import abc
from copy import deepcopy
from functools import reduce
import inspect
import numpy as np

from pysoc.game.game import Game
from pysoc.misc.cache import memoize
from pysoc.misc.prob import normalize_probs


class StrategyParams():
    """A bundle of parameters describing the type of strategy (e.g. complexity criteria)."""
    DEFAULT_PARAMS = {
            'memory_depth': np.inf,  # number of turns into the past that can be used
            'match_data': {'match'},  # the pieces of data that the Strategy uses
            'stochastic': False,  # strategy can use randomness
            'inspects_source': False,  # can analyze strategies
            'manipulates_source': False,  # can alter strategies
            #'manipulates_state': False
        }
    def __init__(self, **kwargs):
        """Initialize from keyword arguments."""
        self.__dict__.update(StrategyParams.DEFAULT_PARAMS)
        self.__dict__.update(kwargs)
        assert (self.inspects_source or (not self.manipulates_source))
    def is_basic(self):
        return (
            (not self.stochastic) and
            (not self.inspects_source) and 
            (not self.manipulates_source) and
            #(not self.manipulates_state) and
            (self.memory_depth in (0, 1))
        )
    def obeys_axelrod(self):
        """Checks whether the strategy obeys Axelrod's original tournament rules."""
        return not (
            self.inspects_source or
            #self.manipulates_state or
            self.manipulates_source
        )
    def __repr__(self):
        return str(self.__dict__)

class Strategy(abc.ABC):
    """A Strategy is a function that, given the current state of a Match, returns an action. The Match will contain information about the game itself as well as all the players' action history. In general, a Strategy can use all of this information, but certain kinds of Strategies may be limited to using only some of this information."""
    PARAMS = StrategyParams()  # bundle of parameters
    VALID_GAME = Game   # a Game subclass for which this strategy is valid
    def set_match(self, index, match):
        """Initialize the persistent data to be used by the strategy. 
        index is the index in the game matrix of the current player using this strategy.
        match is a weakref of a Match object, which may contain state."""
        self.index = index
        self.match = match  # note: weakref avoids strong reference cycle
        # check if the Strategy is valid for the Game
        this_game_cls = match().game.__class__
        if (not issubclass(this_game_cls, self.__class__.VALID_GAME)):
            raise ValueError("Strategy of type {} is invalid for Game of type {}".format(self.__class__, this_game_cls))
    @abc.abstractmethod
    def play_turn(self):
        """Plays one turn of the game. Output is an action (integer)."""
    def __repr__(self):
        def un_array(x):
            """Converts numpy arrays to lists for cleaner printing."""
            if isinstance(x, np.ndarray):
                return '[{}]'.format(', '.join('{:.3f}'.format(elt) for elt in x))
            if (hasattr(x, '__iter__') and (not isinstance(x, str))):
                return [un_array(elt) for elt in x]
            else:
                return x
        args = inspect.getfullargspec(self.__init__).args[1:]
        return "{}({})".format(self.__class__.__name__, ', '.join('{} = {}'.format(arg, un_array(self.__getattribute__(arg))) for arg in args))

class PureStrategy(Strategy):
    """Plays a particular action depending on what player you are."""
    PARAMS = StrategyParams(memory_depth = 0, match_data = {'index'})
    def __init__(self, actions):
        """Initializes from a list of pure actions for each player."""
        self.actions = actions
    def set_match(self, index, match):
        assert (match().game.nplayers == len(self.actions))
        super().set_match(index, match)
    def play_turn(self):
        return self.actions[self.index]

class PureSymmetricStrategy(Strategy):
    """Plays a fixed action no matter what."""
    PARAMS = StrategyParams(memory_depth = 0, match_data = set())
    def __init__(self, action):
        self.action = action 
    def play_turn(self):
        return self.action 

class MixedStrategy(Strategy):
    """For each player, plays one of several pure strategies according to a probability distribution."""
    PARAMS = StrategyParams(memory_depth = 0, match_data = {'index'}, stochastic = True)
    def __init__(self, dists):
        """Initialize from a sequence of probability distributions; the length of each must match the number of the corresponding player's actions."""
        self.dists = [normalize_probs(dist) for dist in dists]
    def set_match(self, index, match):
        assert (match().game.nplayers == len(self.dists))
        assert all(len(dist) == n for (dist, n) in zip(self.dists, match().game.num_actions_by_player))
        super().set_match(index, match)
    def play_turn(self):
        dist = self.dists[self.index]
        return np.random.choice(range(len(dist)), p = dist)

class MixedSymmetricStrategy(Strategy):
    """Plays one of several pure strategies according to a probability distribution, regardless of what player you are."""
    def __init__(self, dist):
        """Initialize from a single probability distribution; the length of each must match the number of all players' actions."""
        self.dist = normalize_probs(dist)
    def set_match(self, index, match):
        assert all(len(self.dist) == n for n in match().game.num_actions_by_player)
        super().set_match(index, match)
    def play_turn(self):
        return np.random.choice(range(len(self.dist)), p = self.dist)

class StrategyMixture(Strategy):
    """Plays one of several arbitrary strategies according to a probability distribution."""
    PARAMS = StrategyParams(stochastic = True)
    def __init__(self, strategies, probs = None):
        """Initialize from a list of strategies and their probabilities. If probs = None, assumes a uniform distribution."""
        self.strategies = strategies
        self.nstrats = len(strategies)
        if (probs is None):
            self.probs = np.ones(self.nstrats) / self.nstrats 
        else:
            assert (len(probs) == self.nstrats)
            self.probs = normalize_probs(probs)
        self.PARAMS.memory_depth = max(strat.PARAMS.memory_depth for strat in self.strategies)
        self.PARAMS.match_data = reduce(set.union, (strat.PARAMS.match_data for strat in self.strategies))
        self.PARAMS.inspects_source = any(strat.PARAMS.inspects_source for strat in self.strategies)
        self.PARAMS.manipulates_source = any(strat.PARAMS.manipulates_source for strat in self.strategies)
    def set_match(self, index, match):
        super().set_match(index, match)
        for strat in self.strategies:
            strat.set_match(index, match)
    def play_turn(self):
        """Choice a strategy randomly accordingly to the strategy probability vector."""
        i = np.random.choice(range(self.nstrats), p = self.probs)
        return self.strategies[i].play_turn()

class ConfusedStrategy(Strategy):
    """Applies to binary games (games in which all players have two actions). When considering any of its opponents' actions, with some probability, the action will be flipped (independently)."""
    def __init__(self, strategy, p):
        self.strategy = strategy 
        assert (0.0 <= p <= 1.0)
        self.p = p 
        self.PARAMS.__dict__.update(self.strategy.PARAMS.__dict__)
        self.PARAMS.stochastic = True
    def set_match(self, index, match):
        assert all(n == 2 for n in match().game.num_actions_by_player)
        super().set_match(index, match)
        self.strategy.set_match(index, match)
    def play_turn(self):
        players = self.match().players
        depth = min(self.PARAMS.memory_depth, len(self.match()))
        if (depth > 0):
            true_actions = [player.action_history.history[-depth:] for player in players]
            # distort the actions (note: we won't bother modifying the counts or the payoffs)
            false_actions = [[action ^ (np.random.rand() <= self.p) for action in row] for row in true_actions]
            for (player, row) in zip(players, false_actions):
                player.action_history.history[-depth:] = row
            action = self.strategy.play_turn()  # compute the action
            # fix the modified actions
            for (player, row) in zip(players, true_actions):
                player.action_history.history[-depth:] = row
        else:
            action = self.strategy.play_turn()
        return action
