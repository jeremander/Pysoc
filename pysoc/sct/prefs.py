from itertools import groupby
import numpy as np
import operator
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import binom

from pysoc.utils import flatten

class PrefRelation(object):
    """Object representation a general preference relation (reflexive, but no requirement of completeness, transitivity, or antisymmetry)."""
    def __init__(self, pref_matrix, items = None):
        """Constructs PrefRelation object from a square Boolean matrix. A True in entry (i, j) indicates i is weakly preferred to j. If False is in entry (j, i), then i is strictly preferred to j."""
        shape = pref_matrix.shape
        if (shape in [(0,), (1, 0)]):
            self.m = 0
        else:
            if ((len(shape) != 2) or (shape[0] != shape[1]) or (pref_matrix.dtype != bool)):
                raise ValueError("Input matrix must be square and Boolean.")
            self.m = shape[0]
        if (items is None):
            self.items = range(self.m)
        else:
            self.items = items[:self.m]  # names of the items
        self.universe = set(self.items)
        if (self.m == 0):
            self.pref_matrix = pref_matrix
        else:
            self.pref_matrix = pref_matrix | np.diag([True for i in range(self.m)]) # Diagonal must be True
        self.item_indices = dict([(self.items[i], i) for i in range(self.m)])
    def prefers(self, x, y, strong = False):
        """Returns True iff x is preferred to y."""
        i = self.item_indices[x]
        j = self.item_indices[y]
        if strong:
            return (self.pref_matrix[i][j] and not(self.pref_matrix[j][i]))
        else:
            return self.pref_matrix[i][j]
    def same_universe_as(self, other):
        """Returns True iff both rankings share the same set of alternatives."""
        return (self.universe == other.universe)
    def partition_by(self, item):
        """Returns list of four lists: [[elts strictly preferred to item], [elts indifferent with item], [elts strictly less preferred to item], [elts incomparable to item]]."""
        partition = [[], [], [], []]
        i = self.item_indices[item]
        for j in range(self.m):
            other = self.items[j]
            if self.pref_matrix[i][j]:
                if self.pref_matrix[j][i]:
                    partition[1].append(other)
                else:
                    partition[2].append(other)
            else:
                if self.pref_matrix[j][i]:
                    partition[0].append(other)
                else:
                    partition[3].append(other)
        return partition
    def is_complete(self, strong = False, verbose = True):
        """Determines whether the preference ordering is complete (weakly or strongly), and provides a counterexample if verbose = True."""
        if isinstance(self, Ranking):
            # Ranking is always weakly complete, and it is strongly so provided there is no indifference
            return ((not strong) or (self.levels == self.m))
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if ((strong and not(self.pref_matrix[i][j] ^ self.pref_matrix[j][i])) or ((not strong) and not(self.pref_matrix[i][j] | self.pref_matrix[j][i]))):
                    if verbose:  # display counterexample
                        sgn = '~' if (self.pref_matrix[i][j] and self.pref_matrix[j][i]) else '?'
                        print("%s %s %s" % (self.items[i], sgn, self.items[j]))
                    return False
        return True
    def is_antisymmetric(self, verbose = True):
        """Determines whether the preference ordering is antisymmetric (i.e. no indifference between distinct items)."""
        if isinstance(self, Ranking):
            return self.is_complete(True, verbose)
        else:
            for i in range(self.m):
                for j in range(i + 1, self.m):
                    if (self.pref_matrix[i][j] and self.pref_matrix[j][i]):
                        if verbose:
                            print("%s ~ %s" % (self.items[i], self.items[j]))
                        return False
            return True
    def is_quasitransitive(self, verbose = True):
        """Determines whether the preference ordering is quasitransitive (i.e. a > b, b > c -> a > c), and provides a counterexample if verbose = True."""
        if (not isinstance(self, Ranking)): # Ranking is always quasitransitive
            # if not a Ranking, we have to do O(m^3) work.
            for i in range(self.m):
                for j in range(self.m):
                    for k in range(self.m):
                        if ((self.pref_matrix[i][j] and not(self.pref_matrix[j][i])) and (self.pref_matrix[j][k] and not(self.pref_matrix[k][j])) and (not(self.pref_matrix[i][k]) or self.pref_matrix[k][i])):
                            if verbose:
                                print("{0} > {1}, {1} > {2}, {0} !> {2}".format(self.items[i], self.items[j], self.items[k]))
                            return False
        return True
    def is_transitive(self, verbose = True):
        """Determines whether the preference ordering is transitive, and provides a counterexample if verbose = True."""
        if (not isinstance(self, Ranking)): # Ranking is always transitive
            # if not a Ranking, we have to do O(m^3) work.
            for i in range(self.m):
                for j in range(self.m):
                    for k in range(self.m):
                        if (self.pref_matrix[i][j] and self.pref_matrix[j][k] and not(self.pref_matrix[i][k])):
                            if verbose:
                                print("{0} >= {1}, {1} >= {2}, {0} !>= {2}".format(self.items[i], self.items[j], self.items[k]))
                            return False
        return True
    def get_ranking(self):
        """If the PrefRelation object is weakly complete and transitive, convert it to a Ranking object."""
        if (self.m == 0):
            return Ranking([])
        if not(self.is_complete() and self.is_transitive()):
            raise ValueError("The relation is not a complete preorder.")
        maxima = self.maximum_elements()
        remainder = self.reduce_to_subset([item for item in self.items if (item not in maxima)])
        return Ranking(maxima + remainder.get_ranking().items)
    def reverse(self):
        """Returns new PrefRelation object such that for all pairs (x, y) such that x > y, the ordering is reversed."""
        pref_matrix = np.array(self.pref_matrix)
        for i in range(self.m):
            for j in range(i + 1, self.m):
                if (self.pref_matrix[i][j] ^ self.pref_matrix[j][i]):
                    pref_matrix[i][j] = not pref_matrix[i][j]
                    pref_matrix[j][i] = not pref_matrix[j][i]
        return PrefRelation(pref_matrix, self.items)
    def _extremal_elements(self, maximal):
        """Returns the set of maximal elements or minimal elements, depending on if maximal = True."""
        elts = []
        for i in range(self.m):
            is_extremal = True
            for j in range(self.m):
                if ((maximal and self.pref_matrix[j][i] and not(self.pref_matrix[i][j])) or \
                    ((not maximal) and self.pref_matrix[i][j] and not(self.pref_matrix[j][i]))):
                    is_extremal = False
                    break
            if is_extremal:
                elts.append(self.items[i])
        return elts
    def maximal_elements(self):
        """Returns the set of maximal elements (that is, {x | no y > x})."""
        return self._extremal_elements(True)
    def minimal_elements(self):
        """Returns the set of minimal elements (that is, {x | no y < x})."""
        return self._extremal_elements(False)
    def _extremum_elements(self, maximum):
        """Returns the set of maximum or minimum elements, depending on if maximum = True."""
        elts = []
        for i in range(self.m):
            is_extremum = True
            for j in range(self.m):
                if ((maximum and not(self.pref_matrix[i][j])) or \
                    ((not maximum) and not(self.pref_matrix[j][i]))):
                    is_extremum = False
                    break
            if is_extremum:
                elts.append(self.items[i])
        return elts
    def maximum_elements(self):
        """Returns the set of maximum elements (that is, {x | all y <= x})."""
        return self._extremum_elements(True)
    def minimum_elements(self):
        """Returns the set of minimum elements (that is, {x | all y >= x})."""
        return self._extremum_elements(False)
    def max(self):
        """Returns the unique greatest element if there is one, otherwise None."""
        maxima = self.maximum_elements()
        if (len(maxima) != 1):
            return None
        return maxima[0]
    def min(self):
        """Returns the unique least element if there is one, otherwise None."""
        minima = self.minimum_elements()
        if (len(minima) != 1):
            return None
        return minima[0]
    def reduce_to_subset(self, subset):
        """Given a subset of the candidates, reduces the PrefOrdering to this subset."""
        if (len(subset) == 0):
            return PrefRelation(np.array([]), [])
        subset_indices = [self.item_indices[item] for item in subset]
        mat = self.pref_matrix[subset_indices, :][:, subset_indices]
        items = [self.items[i] for i in subset_indices]
        return PrefRelation(mat, items)
    def __repr__(self):
        return repr(pd.DataFrame(self.pref_matrix, index = self.items, columns = self.items))
    def __len__(self):
        return self.m
    @classmethod
    def random(cls, items):
        if isinstance(items, int):
            items = range(items)
        pref_matrix = (np.random.randint(0, 2, [len(items), len(items)]) > 0)
        return cls(pref_matrix, items)

class Ranking(PrefRelation):
    """Object representing a preference ranking (complete, reflexive, transitive, set of ranked preferences, i.e. a complete preorder.)."""
    def __init__(self, items):
        """Constructor with list of ranked items. If a list of items appears in the list instead of a single item, this indicates indifference between the items in the sublist."""
        self.universe = flatten(items)
        if (len(set(self.universe)) != len(self.universe)):
            raise ValueError("Members of ranked item list must be unique.")
        self.universe = set(self.universe)
        self.m = len(self.universe)  # number of alternatives
        self.items = []
        for item in items:  # put into canonical form (list of lists)
            if isinstance(item, list):
                self.items.append(item)
            else:
                self.items.append([item])
        self.levels = len(self.items)  # number of distinct rankings
        self.item_ranks = dict()
        for i in range(self.levels):
            for elt in self.items[i]:
                self.item_ranks[elt] = i
    def is_strict(self):
        return all(len(group) == 1 for group in self.items)
    def to_strict(self):
        return StrictRanking([group[0] for group in self.items])
    def rank(self, item):
        """Returns the 0-up rank of a given item."""
        return self.item_ranks[item]
    def prefers(self, x, y, strong = False):
        """Returns True iff x is preferred to y."""
        if strong:
            return self.rank(x) < self.rank(y)
        else:
            return self.rank(x) <= self.rank(y)
    def reverse(self):
        """Returns new Ranking object such that the ordering is reversed."""
        items = list(self.items)
        items.reverse()
        return Ranking(items)
    def partition_by(self, item):
        """Returns list of four lists: [[elts strictly preferred to item], [elts indifferent with item], [elts strictly less preferred to item], [elts incomparable to item]]."""
        r = self.rank(item)
        return [flatten([self[i] for i in range(r)]), self[r], flatten([self[i] for i in range(r + 1, self.levels)]), []]
    def _extremal_elements(self, maximal):
        """Returns the set of maximal elements or minimal elements, depending on if maximal = True."""
        if (self.m == 0):
            return []
        if maximal:
            return self.items[0]
        return self.items[-1]
    def _extremum_elements(self, maximum):
        """Returns the set of maximum or minimum elements, depending on if maximum = True."""
        return self._extremal_elements(maximum)  # coincide with extremal elements for complete preferences
    def reduce_to_subset(self, subset):
        """Given a subset of the candidates, reduces the Ranking to this subset."""
        items = [[x for x in lst if (x in subset)] for lst in self.items]
        items = [lst for lst in items if (len(lst) > 0)]
        return Ranking(items)
    def get_ranking(self):
        return self
    def break_ties_randomly(self) -> 'StrictRanking':
        """For each indifference class, breaks ties randomly, and returns a StrictRanking."""
        return StrictRanking([x for xs in self.items for x in np.random.permutation(xs)])
    def __repr__(self):
        s = ""
        for i in range(self.levels):
            for j in range(len(self[i])):
                s += "%s" % str(self[i][j])
                if (j < len(self[i]) - 1):
                    s += ", "
            s += "\n"
        return s
    def __getitem__(self, index):
        return self.items[index]
    @classmethod
    def random(cls, items, indifference_prob = 0.0):
        """Generates random ranking among some items. indifference_prob is the probability that two adjacently ranked items are in fact indifferent to each other."""
        if isinstance(items, int):
            items = range(items)
        n = len(items)
        perm = np.random.permutation(n)
        items_list = []
        current_list = []
        for i in range(n):
            current_list.append(items[perm[i]])
            if ((np.random.rand() >= indifference_prob) or (i == n - 1)):
                items_list.append(current_list)
                current_list = []
        return cls(items_list)

class StrictRanking(Ranking):
    """Object representing a strict preference ranking (complete, reflexive, transitive, antisymmetric set of ranked preferences, i.e. a complete order)."""
    def __init__(self, items):
        """Constructor with list of ranked items."""
        self.universe = items
        if (len(set(self.universe)) != len(self.universe)):
            raise ValueError("Members of ranked item list must be unique.")
        self.universe = set(self.universe)
        self.m = len(self.universe)  # number of alternatives
        self.items = items
        self.item_ranks = {item : i for (i, item) in enumerate(self.items)}
    def is_strict(self):
        return True
    def to_strict(self):
        return self
    def partition_by(self, item):
        """Returns list of four lists: [[elts strictly preferred to item], [elts indifferent with item], [elts strictly less preferred to item], [elts incomparable to item]]."""
        r = self.rank(item)
        return [[self[i] for i in range(r)], [self[r]], [self[i] for i in range(r + 1, self.m)], []]
    def _extremal_elements(self, maximal):
        """Returns the set of maximal elements or minimal elements, depending on if maximal = True."""
        if (self.m == 0):
            return []
        if maximal:
            return [self.items[0]]
        return [self.items[-1]]
    def reduce_to_subset(self, subset):
        """Given a subset of the candidates, reduces the StrictRanking to this subset."""
        items = [x for x in self.items if x in subset]
        return StrictRanking(items)
    def break_ties_randomly(self) -> 'StrictRanking':
        return self
    def __repr__(self):
        return ', '.join(str(x) for x in self)
    @classmethod
    def random(cls, items):
        """Generates random ranking among some items."""
        if isinstance(items, int):
            items = range(items)
        perm = np.random.permutation(len(items))
        items_list = [items[i] for i in perm]
        return cls(items_list)

def kendall_tau(rank1, rank2):
    """Computes (reliable) Kendall tau correlation between two StrictRankings."""
    n = len(rank1)
    assert(len(rank2) == n)
    r1 = rank1.items
    r2 = rank2.items
    discordant_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ((r1.index(i) > r1.index(j)) != (r2.index(i) > r2.index(j))):
                discordant_pairs += 1
    denom = binom(n, 2)
    return ((denom - 2 * discordant_pairs) / denom)

class CardinalRanking(Ranking):
    def __init__(self, score_dict):
        """Takes a dictionary mapping items to scores (real numbers), and constructs an ordinal ranking based on these scores, while retaining the cardinal score information."""
        assert(isinstance(score_dict, dict))
        self.score_dict = score_dict
        items_and_scores = list(self.score_dict.items())
        items_and_scores.sort(key = operator.itemgetter(1), reverse = True)
        self.scores = []
        ranking = []
        for (score, group) in groupby(items_and_scores, operator.itemgetter(1)):
            self.scores.append(score)
            ranking.append([pair[0] for pair in group])
        super().__init__(ranking)
        try:
            self.item_width = max([len(str(item)) for item in self.items])  # for display purposes
        except ValueError:
            breakpoint()
    def score(self, item):
        """Returns score of a given item."""
        return self.score_dict[item]
    def plot(self):
        """Plot the histogram of scores."""
        plt.clf()
        plt.figure(1)
        plt.hist(self.scores, max(10, int(np.round(np.sqrt(self.m)))))
        plt.xlabel('Scores')
        xlim = 1.1 * max([abs(score) for score in self.scores])
        plt.xlim((-xlim, xlim))
        plt.show()
    def __repr__(self):
        s = ''
        for (score, group) in zip(self.scores, self.items):
            group_str = '[{}]'.format(', '.join(str(x) for x in group))
            s += f'{score}: {group_str}\n'
        return s
    def __add__(self, other):
        """Add scores with another CardinalRanking object (or a constant) to obtain new ranking."""
        sd = dict(self.score_dict)
        if isinstance(other, (int, float)):
            for item in self.universe:
                sd[item] += other
        else:
            if (self.universe != other.universe):
                raise ValueError("Domains must be equal in order to add.")
            for item in self.universe:
                sd[item] += other.score_dict[item]
        return CardinalRanking(sd)
    def __radd__(self, other):
        return self + other
    def __mul__(self, c):
        """Multiply all scores by a constant."""
        sd = dict(self.score_dict)
        for item in self.universe:
            sd[item] *= c
        return CardinalRanking(sd)
    def __rmul__(self, c):
        return self * c
    def __div__(self, c):
        """Divide all scores by a constant."""
        sd = dict(self.score_dict)
        for item in self.universe:
            sd[item] /= c
        return CardinalRanking(sd)
    def __rdiv__(self, c):
        return self / c
    @classmethod
    def gaussian(cls, items, mu = 0.0, sigma = 1.0):
        """Generates scores as i.i.d. Gaussian random variables."""
        if isinstance(items, int):
            items = range(items)
        score_dict = dict()
        for item in items:
            score_dict[item] = random.gauss(mu, sigma)
        return cls(score_dict)
    @classmethod
    def uniform(cls, items, min_score = -1.0, max_score = 1.0):
        """Generates scores as i.i.d. uniform random variables."""
        if isinstance(items, int):
            items = range(items)
        score_dict = dict.fromkeys(items)
        scale = max_score - min_score
        for item in items:
            score_dict[item] = min_score + random.random() * scale
        return cls(score_dict)
    @classmethod
    def beta(cls, items, a, b):
        """Generates scores as i.i.d. Beta(a, b) random variables."""
        if isinstance(items, int):
            items = range(items)
        score_dict = dict.fromkeys(items)
        for item in items:
            score_dict[item] = beta.rvs(a, b)
        return cls(score_dict)

class Profile(object):
    """Object representing a vector of ranked preferences."""
    def __init__(self, prefs, names = None):
        """Constructor takes list of PrefRelation objects, and (optionally) an identically-sized list of names of the population members. Default will be range(n), where n = len(prefs)."""
        self.prefs = prefs
        self.n = len(self.prefs)
        if (self.n > 0):
            self.universe = self.prefs[0].universe
            for i in range(1, self.n):
                if (self.prefs[i].universe != self.universe):
                    raise ValueError("All rankings must have the same set of alternatives.")
        else:
            self.universe = set()
        self.universe_list = list(self.universe)  # handy to have list representation around too
        self.m = len(self.universe)
        if (names is None):
            self.names = range(self.n)
        else:
            assert (len(names) == self.n), "Mismatch between number of PrefRelations and number of names."
            self.names = names[:self.n]
        self.indices_by_name = {name : i for (i, name) in enumerate(self.names)}
    def reduce_to_subset(self, subset):
        """Given a subset of the candidates, reduces each member's PrefRelation to this subset."""
        prefs = [self.prefs[i].reduce_to_subset(subset) for i in range(self.n)]
        return Profile(prefs, self.names)
    def reverse(self):
        """For each member in the Profile, reverse the ordering of their preferences."""
        return Profile([self.prefs[i].reverse() for i in range(self.n)], self.names)
    def all_prefs_ranked(self):
        """Returns True if all members' preferences are Ranking objects (complete preorders)."""
        return all(isinstance(pref, Ranking) for pref in self.prefs)
    def members_preferring(self, x, y, strong = False):
        """Returns list of individuals (strongly) preferring x to y."""
        return [self.names[i] for i in range(self.n) if self.prefs[i].prefers(x, y, strong)]
    def number_preferring(self, x, y, strong = False):
        """Returns number of individuals (strongly) preferring x to y."""
        return len(self.members_preferring(x, y, strong))
    def _condorcet_set(self, strong, cmp_fun):
        """Returns set of (strong or weak) Condorcet winners or losers. For x to be a weak winner, the number weakly preferring x to y must be at least the number weakly preferring y to x, for all y != x. For x to be a strong winner, the number strongly preferring x to y must be greater than the number strongly preferring y to x, for all y != x. Similar definitions apply for losers."""
        condorcet_set = []
        for i in range(self.m):
            is_condorcet = True
            x = self.universe_list[i]
            for j in range(self.m):
                y = self.universe_list[j]
                if ((i != j) and cmp_fun(self.number_preferring(x, y, strong), self.number_preferring(y, x, strong))):
                    is_condorcet = False
                    break
            if is_condorcet:
                condorcet_set.append(x)
        return condorcet_set
    def condorcet_winners(self, strong = False):
        """Returns set of (strong or weak) Condorcet winners."""
        cmp_fun = operator.le if strong else operator.lt
        return self._condorcet_set(strong, cmp_fun)
    def condorcet_losers(self, strong = False):
        """Returns set of (strong or weak) Condorcet losers."""
        cmp_fun = operator.ge if strong else operator.gt
        return self._condorcet_set(strong, cmp_fun)
    def ballot(self, split = True, subset = None):
        """Returns dictionary of items with number of votes each obtains. A member votes for an item if it is his/her top choice. If split = False, then the function is undefined if a member does not have a single top-ranked item, in which case an exception is raised. Otherwise, will take maxima and split one vote amongst them. Optional argument subset takes a list that is a subset of candidates to consider in the ballot."""
        if (subset is None):
            ballot = dict.fromkeys(self.universe_list, 0)
        else:
            ballot = dict.fromkeys(subset, 0)
        for i in range(self.n):
            prefs = self.prefs[i] if (subset is None) else self.prefs[i].reduce_to_subset(subset)
            maximum = prefs.max()
            if (maximum is None):
                if split:
                    maxima = prefs.maximum_elements()
                    if (len(maxima) > 0):
                        for item in maxima:
                            ballot[item] += 1.0 / len(maxima)
                    else:
                        raise ValueError("All voters must have top-ranked preferences.")
                else:
                    raise ValueError("All voters must have a top-ranked preference.")
            else:
                ballot[maximum] += 1
        return ballot
    def __repr__(self):
        s = ""
        for (name, rel) in zip(self.names, self.prefs):
            if isinstance(rel, Ranking):
                if rel.is_strict():
                    s += f'{name} : {rel.to_strict()!r}\n'
                else:
                    s += "{} : [{}]\n".format(name, ', '.join('[{}]'.format(', '.join(str(x) for x in group)) for group in rel.items))
            else:
                s += "{} : <PrefRanking on \{{}\}\n".format(name, repr(rel.items)[1:-1])
        return s
    def __iter__(self):
        return iter(self.prefs)
    def __getitem__(self, name):
        # try:
        #     name = int(name)
        # except ValueError:
        #     name = name
        return self.prefs[self.indices_by_name[name]]
            # return self.prefs[self.indices_by_name[name]]
        # except KeyError:
            # breakpoint()
    def to_csv(self, filename):
        num_items = len(self.universe)
        with open(filename, 'w') as f:
            f.write(','.join([''] + [str(i) for i in range(num_items)]) + '\n')
            for (name, ranking) in zip(self.names, self.prefs):
                f.write(str(name))
                for (i, group) in enumerate(ranking):
                    f.write(',' + ';'.join(group))
                for _ in range(i + 1, num_items):
                    f.write(',')
                f.write('\n')
    @classmethod
    def from_csv(cls, filename):
        df = pd.read_csv(filename, index_col = 0, dtype = str).fillna('')
        numeric_cols = sorted([col for col in df.columns if all(c.isdigit() for c in col)], key = int)
        rankings = []
        for tup in df[numeric_cols].itertuples():
            ranking = []
            for col in tup[1:]:
                if (len(col) == 0):
                    break
                else:
                    ranking.append(col.split(';'))
            rankings.append(Ranking(ranking))
        for ranking in rankings:
            if ranking.universe != rankings[0].universe:
                breakpoint()
        assert all(ranking.universe == rankings[0].universe for ranking in rankings)
        return cls(rankings, names = list(df.index))
    @classmethod
    def random(cls, items, n = None, names = None, indifference_prob = 0.0):
        """Generates random profile among some items (each member has ranked preferences, with given probability of indifference)."""
        assert (0.0 <= indifference_prob <= 1.0)
        if isinstance(items, int):
            items = range(items)
        else:
            items = list(items)
        if (n is None):
            if (names is None):
                n = 1
            else:
                n = len(names)
        prefs = []
        for i in range(n):
            if (indifference_prob == 0.0):
                prefs.append(StrictRanking.random(items))
            else:
                prefs.append(Ranking.random(items, indifference_prob))
        return cls(prefs, names)


# Carols (Cryptones 2014)
carols = Profile([Ranking([['God', 'Hark', 'Holly', 'What'], 'Deck']), \
                  Ranking(['Deck', 'Hark', 'God', 'What', 'Holly']), \
                  Ranking(['What', 'God', ['Deck', 'Hark'], 'Holly']), \
                  Ranking([['Hark', 'What'], 'Holly', 'God', 'Deck']), \
                  Ranking([['Holly', 'God', 'Hark'], 'What', 'Deck']), \
                  Ranking(['Hark', 'Deck', 'Holly', 'What', 'God']), \
                  Ranking(['Holly', 'What', 'Hark', 'God', 'Deck']), \
                  Ranking(['Deck', ['What', 'Hark'], 'God', 'Holly']), \
                  Ranking(['God', 'Hark', 'Holly', 'Deck', 'What'])])
