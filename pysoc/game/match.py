from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import weakref

from .player import Player

def force_integer_xaxis():
    plt.axes().xaxis.set_major_locator(MaxNLocator(integer = True))

class ActionHistory():
    """A list of actions for a player, along with counts of actions played thus far."""
    def __init__(self, action_names):
        """action_names is a sequence of names, giving a correspondence between integers and names."""
        self.action_names = action_names
        self.action_indices_by_name = {action : i for (i, action) in enumerate(action_names)}
        self.num_actions = len(action_names)
        self.clear()
    def clear(self):
        """Clears the action history."""
        self.action_counts = np.zeros(self.num_actions, dtype = int)        
        self.history = []
    def _update_one(self, action):
        """Adds a new action to the history."""
        if isinstance(action, str):
            action = self.action_indices_by_name[action]
        self.action_counts[action] += 1
        self.history.append(action)
    def _update_many(self, actions):
        """Adds a sequence of actions to the history."""
        for action in actions:
            self._update_one(action)
    def update(self, actions):
        """Adds a new action or actions to the history."""
        try:
            self._update_many(actions)
        except:
            self._update_one(actions)
    def __len__(self):
        return len(self.history)
    def __iter__(self):
        return iter(map(lambda i : self.action_names[i], self.history))
    def __getitem__(self, i):
        return self.history[i]
    def __repr__(self):
        counts = pd.Series(self.action_counts, index = self.action_names)
        s = 'History\n'
        s += ' '.join(self.action_names[i] for i in self.history) + '\n'
        if (len(self.history) > 0):
            s += '\n'
        s += 'Counts\n'
        s += '\n'.join(repr(counts).split('\n')[:-1])
        return s

class PayoffHistory():
    """A list of payoffs for a player."""
    def __init__(self):
        self.clear()
    def clear(self):
        """Clears the payoff history."""
        self.payoffs = []
        self.cumulative_payoffs = [0]
        self.total_payoff = 0
    def _update_one(self, payoff):
        """Adds a new payoff to the history."""
        self.total_payoff += payoff
        self.payoffs.append(payoff)
        self.cumulative_payoffs.append(self.total_payoff)        
    def _update_many(self, payoffs):
        """Adds a sequence of payoffs to the history."""
        for payoff in payoffs:
            self._update_one(payoff)
    def update(self, payoffs):
        """Adds a new payoff or payoff to the history."""
        try:
            self._update_many(payoffs)
        except:
            self._update_one(payoffs)
    def plot(self, show = True):
        """Plots a graph of the payoffs over time."""
        force_integer_xaxis()
        p = plt.plot(self.payoffs, marker = 'o', markersize = 8, alpha = 0.7)
        if show:
            plt.show()
        return p
    def plot_cumulative(self, show = True):
        """Plots a graph of the cumulative payoff over time."""
        force_integer_xaxis()
        p = plt.plot(self.cumulative_payoffs, marker = 'o', markersize = 8, alpha = 0.7)
        if show:
            plt.show()
        return p
    def __len__(self):
        return len(self.payoffs)
    def __iter__(self):
        return iter(self.payoffs)
    def __getitem__(self, i):
        return self.payoffs[i]
    def __repr__(self):
        return repr(self.payoffs)

class Match():
    """A sequence of actions taken by all players in a Game, yielding payoffs at each turn."""
    def __init__(self, game, players = None, nturns = None, p_end = None):
        """game is a Game to be played.
           players is a list of Players playing the Game.
           nturns is the fixed number of turns in the Match.
           p_end is the probability the game terminates after each play."""
        self.game = game
        if (players is None):  # give dummy names to the Players
            self.players = [Player(str(i), None) for i in range(self.game.nplayers)]
        else:
            assert (len(players) == self.game.nplayers)
            self.players = players
        self.nturns = nturns
        assert (p_end is None) or (0.0 <= p_end <= 1.0)
        self.p_end = p_end
        self._has_finite_turns = (self.nturns is not None) and np.isfinite(self.nturns)
        self._terminates_randomly = (self.p_end is not None) and (self.p_end > 0.0)
        self.clear()
    def clear(self):
        """Clears the match history."""
        # reset action histories and payoff histories for each player
        for (player, action_names) in zip(self.players, self.game.action_names):
            player.action_history = ActionHistory(action_names)
            player.payoff_history = PayoffHistory()
        self.turn = 0
        self._terminated = False  # flag for probabilistic termination
    def set_strategies(self, strategies):
        """Sets the Strategy for each player."""
        assert (len(strategies) == self.game.nplayers)
        for (i, (strategy, player)) in enumerate(zip(strategies, self.players)):
            player.strategy = strategy
            player.strategy.set_match(i, weakref.ref(self))  # avoid strong reference cycle
    @property
    def terminated(self):
        return self._terminated
    def _update_one(self, actions):
        """Given a sequence of actions for each player, adds these actions to each player's action history, then computes the payoffs and adds these to each player's payoff history."""
        payoffs = self.game.payoff(actions)
        for (action, payoff, player) in zip(actions, payoffs, self.players):
            player.action_history._update_one(action)
            player.payoff_history._update_one(payoff)
        self.turn += 1
        if (self._has_finite_turns and (self.turn >= self.nturns)):  # terminate at max turns
            self._terminated = True
        # randomly terminate
        if (self._terminates_randomly and (np.random.rand() <= self.p_end)):  # terminate randomly
            self._terminated = True
    def _update_many(self, actionss):
        """Input is a sequence of action sequences for each player."""
        for actions in actionss:
            if self._terminated:
                print(self.termination_message())
                break
            self._update_one(actions)
    def update(self, actions):
        """Given a sequence of actions for each player, adds these actions to each player's action history, then computes the payoffs and adds these to each player's payoff history."""
        try:
            self._update_many(actions)
        except:
            self._update_one(actions)
    def is_terminable(self):
        """Returns True if the Match can terminate eventually."""
        return (self._has_finite_turns or self._terminates_randomly)
    def termination_message(self):
        """Returns string to be printed when the termination condition has been reached."""
        return "Termination condition reached after {} turns.".format(self.turn)
    def _play(self):
        actions = [player.strategy.play_turn() for player in self.players]
        self._update_one(actions)           
    def play(self, nturns = 1):
        """Plays one or more turns of a match. All of the players are assumed to have strategies that are played against one another to produce actions."""
        for j in range(nturns):
            if self._terminated:
                print(self.termination_message())    
                break        
            self._play()
    def play_to_completion(self):
        """Plays the Match until termination is reached."""
        if (not self.is_terminable()):
            raise RuntimeError("Cannot play Match to completion unless a finite termination condition exists.")             
        while (not self._terminated):
            self._play()
    def plot_payoffs(self, show = True):
        """Plots a graph of all players' payoffs over time."""
        lines = [player.payoff_history.plot(show = False)[0] for player in self.players]
        plt.legend(lines, map(str, self.players))
        plt.title('Payoffs for Game "{}"\n{} turns'.format(self.game.name, self.turn), fontweight = 'bold', fontsize = 12)
        force_integer_xaxis()
        if show:
            plt.show()
    def plot_cumulative_payoffs(self, show = True):
        """Plots a graph of all players' cumulative payoffs over time."""
        lines = [player.payoff_history.plot_cumulative(show = False)[0] for player in self.players]
        plt.legend(lines, map(str, self.players))
        plt.title('Cumulative payoffs for Game "{}"\n{} turns'.format(self.game.name, self.nturns), fontweight = 'bold', fontsize = 12)
        force_integer_xaxis()
        if show:
            plt.show()
    def outcome_counts(self):
        """Gets dictionary mapping each outcome to the number of times it occurs."""
        return Counter(zip(*(player.action_history.history for player in self.players)))
    def outcome_table(self):
        """Counts the number of times each action configuration is played, and returns a table of these. If it is a 2-player game, return it as a DataFrame with action labels."""
        outcome_counts = self.outcome_counts()
        game = self.game
        mat = np.zeros(game.num_actions_by_player, dtype = int)
        for (outcome, count) in outcome_counts.items():
            mat[outcome] = count
        if (game.nplayers == 2):
            mat = pd.DataFrame(mat, index = game.action_names[0], columns = game.action_names[1])
        return mat
    def winners(self):
        """Returns list of player indices with the maximum total payoff."""
        max_payoff = max(player.payoff_history.total_payoff for player in self.players)
        return [i for (i, player) in enumerate(self.players) if (player.payoff_history.total_payoff == max_payoff)]
    # TODO: convert scores to rankings (weak total orderings)
    def view_history(self, symbols = None, width = 50):
        """Displays the history concisely.
        symbols is a sequence of symbols to be used for each action.
        width specifies the number of turns to show on each line."""
        player_name_width = max(len(player.name) for player in self.players)
        num_actions = max(self.game.num_actions_by_player)
        if (symbols is None):
            if (num_actions <= 2):
                symbols = ['·', '█']
            elif (num_actions <= 10):
                symbols = list(map(str, range(num_actions)))
            else:
                raise ValueError("Cannot concisely view more than 10 actions.")
        else:
            assert (len(symbols) >= num_actions)
        s = ''
        actions = list(zip(*(player.action_history.history for player in self.players)))
        lines = [zip(*actions[width * i : width * (i + 1)]) for i in range(int(np.ceil(len(actions) / width)))]
        for line in lines:
            for (i, (player, playerline)) in enumerate(zip(self.players, line)):
                s += '{}: '.format(player.name.rjust(player_name_width))
                s += ''.join(symbols[action] for action in playerline) + '\n'
            s += '\n'
        print(s)
    def __repr__(self):
        s = repr(self.game)
        s += '\n\nPlayers:\n' + '\n'.join(repr(player) for player in self.players)
        s += "\n\nMatch on turn {}\n\n".format(len(self))
        if (len(self) > 0):
            index = []
            table = []
            for player in self.players:
                index += [str(player.name), '', '']
                table.append(list(player.action_history))
                table.append(list(player.payoff_history))
                table.append(list(player.payoff_history.cumulative_payoffs[1:]))
            df = pd.DataFrame(table, index = index)
            lines = str(df).split('\n')[1:]
            s += '\n'.join('\n'.join(lines[3 * i : 3 * (i + 1)]) + '\n' for i in range(self.game.nplayers))
        return s
    def __len__(self):
        return self.turn

class NoisyMatch(Match):
    """A Match (for 2-action games only) where there is some positive probability of each player's action being flipped."""
    def __init__(self, game, players = None, nturns = None, p_end = None, p_noise = 0.1):
        """p_noise is either a single probability (applied to all players) or a sequence of probabilities for each player."""
        super().__init__(game, players = players, nturns = nturns, p_end = p_end)
        if (not all(num_actions == 2 for num_actions in self.game.num_actions_by_player)):
            raise ValueError("All players must have exactly 2 actions to have a noisy match.")
        if (not hasattr(p_noise, '__len__')):
            p_noise = [p_noise] * self.game.nplayers
        assert all(0.0 <= p <= 1.0 for p in p_noise)
        self.p_noise = p_noise
    def _play(self):
        actions = [player.strategy.play_turn() ^ (np.random.rand() <= p) for (player, p) in zip(self.players, self.p_noise)]
        self._update_one(actions)  


