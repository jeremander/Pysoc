from collections import defaultdict
import itertools
from math import factorial
import numpy as np
from tqdm import tqdm
from typing import List

from pysoc.sct.prefs import CardinalRanking, PrefRelation, Profile, Ranking, StrictRanking

# Social Welfare Functions
# (AKA preference aggregation)
# Take Profiles as input, return (set of) PrefRelations as output.

def pairwise_majority_vote(profile: Profile) -> PrefRelation:
    """Determines for each pair (x, y) whether x >= y in the output relation based on the criterion N(x >= y) >= N(y >= x), where N(x >= y) is the number of members weakly preferring x to y."""
    pref_matrix = np.zeros((profile.m, profile.m), dtype = bool)
    for i in range(profile.m):
        for j in range(profile.m):
            pref_matrix[i][j] = (i == j) or (profile.number_preferring(profile.universe_list[i], profile.universe_list[j], False) >= profile.number_preferring(profile.universe_list[j], profile.universe_list[i], False))
    return PrefRelation(pref_matrix, profile.universe_list)

def kemeny_young(profile: Profile) -> List[Ranking]:
    """Create matrix of tallies (number strongly preferring x to y). Then score each ranking by adding up pairwise tallies."""
    # First create m x m matrix of total number strongly preferring i to j
    tallies = np.zeros((profile.m, profile.m))
    for i in range(profile.m):
        for j in range(profile.m):
            if (i != j):
                tallies[i][j] = profile.number_preferring(profile.universe_list[i], profile.universe_list[j], True)
    # Now assign scores to each permutation
    best_perms = []
    best_score = -1
    perms = tqdm(itertools.permutations(range(profile.m)), total = factorial(profile.m))
    for perm in perms:
        score = 0
        for ii in range(profile.m):
            for jj in range(ii + 1, profile.m):
                (i, j) = (perm[ii], perm[jj])
                score += tallies[i][j]
        if (score > best_score):
            best_perms = [perm]
            best_score = score
        elif (score == best_score):
            best_perms.append(perm)
    best_rankings = [Ranking([profile.universe_list[i] for i in perm]) for perm in best_perms]
    return best_rankings

def borda_count_ranking(profile: Profile) -> Ranking:
    """In strict linear ordering, each person contributes a score of (m - k - 1) to the item ranked k (0-up). E.g. if 3 items, gives 2 to first-ranked, 1 to second-ranked, 0 to last-ranked item. Here we modify the rule to allow for indifferent preferences. Order the indifferent ones arbitrarily, but instead of assigning the usual score to each of these, average the scores within the equivalence class."""
    if (not profile.all_prefs_ranked()):
        raise ValueError("All preferences must be complete preorders.")
    borda_counts = dict.fromkeys(profile.universe_list, 0.0)
    for i in range(profile.n):
        if isinstance(profile.prefs[i], StrictRanking):  # convert to a weak ranking
            prefs = [[x] for x in profile.prefs[i].items]
        else:
            prefs = profile.prefs[i].items
        count = profile.m - 1
        for equiv_class in prefs:
            size = len(equiv_class)
            class_score = count - (size - 1) / 2.0
            for item in equiv_class:
                borda_counts[item] += class_score
            count -= size
    return CardinalRanking(borda_counts)
    # pairs = sorted(borda_counts.items(), key = itemgetter(1), reverse = True)
    # ranking = []
    # equiv_class = []
    # for (item, ct) in pairs:
    #     if (not equiv_class):
    #         equiv_class.append((item, ct))
    #     elif (ct < equiv_class[-1][0]):
    #         ranking.append(equiv_class)
    #         equiv_class = [(item, ct)]
    # if equiv_class:
    #     ranking.append(equiv_class)
    # return Ranking(ranking)


# Social Choice Functions
# Take Profiles as input, return set of winners as output.

class SCFCollection():
    """Represents a collection of SCFs. Can call all of these on a profile to readily compare methods."""
    def __init__(self, scfs):
        """scf is a list of SCFs. Each SCF is assumed to take just a Profile as an argument."""
        self.scfs = scfs
    def __call__(self, profile):
        """Calls all the SCFs on a Profile, returning the results as a dict."""
        return {repr(scf) : scf(profile) for scf in self.scfs}
    def report_all(self, profile):
        """Calls all the SCFs on a Profile, returning the results as a string."""
        s = ''
        for scf in self.scfs:
            s += '\n{!r}:\n'.format(scf)
            s += '\t[{}]\n'.format(', '.join(str(x) for x in sorted(scf(profile))))
        return s
    def __iter__(self):
        return iter(self.scfs)
    def __len__(self):
        return len(self.scfs)
    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.scfs)

# global registry of SCFs
SCF_COLLECTION = SCFCollection([])

def scf(f):
    """Decorates a social choice function with simple handling of trivial cases."""
    class WrappedSCF():
        def __call__(self, profile, *args, **kwargs):
            if (profile.n == 0):
                raise ValueError("Must have nonzero number of voters.")
            if (profile.m == 0):
                return []
            if (profile.m == 1):
                return [profile.universe_list[0]]
            return f(profile, *args, **kwargs)
        def __repr__(self):
            return f.__name__
    wrapped = WrappedSCF()
    SCF_COLLECTION.scfs.append(wrapped)
    return wrapped

@scf
def weak_condorcet_winners(profile):
    return profile.condorcet_winners(strong = False)

@scf
def strong_condorcet_winners(profile):
    return profile.condorcet_winners(strong = True)

@scf
def majority_vote(profile, split = True):
    """If there is a majority (>= 50%) that ranks x as its top choice, output set is [x]. If no majority exists, output set is []. If two choices x and y attain majorities, output set is [x, y]. If split = False, requires each PrefRelation to have a unique maximum; otherwise, splits the ballot equally across the maxima."""
    ballot = profile.ballot(split)
    winners = []
    for item in ballot:
        if (float(ballot[item]) >= profile.n / 2.0):
            winners.append(item)
    return winners

@scf
def plurality_vote(profile, split = True):
    """Returns the items that the largest number of members rank as the highest (ties permitted). If split = False, requires each PrefRelation to have a unique maximum; otherwise, splits the ballot equally across the maxima."""
    ballot = profile.ballot(split)
    max_votes = max(ballot.values())
    winners = [item for item in ballot if ballot[item] == max_votes]
    return winners

@scf
def plurality_with_runoff(profile, split = True):
    """If the majority winner set is non-empty, returns that. Otherwise, does a plurality vote to get the top (at least two) candidates, then does a majority contest between these."""
    maj_winners = majority_vote(profile)
    if (len(maj_winners) > 0):
        return maj_winners
    ballot = profile.ballot(split)
    votes = sorted(ballot.values())
    votes.reverse()
    max_votes = max(ballot.values())
    if (votes.count(max_votes) >= 2):
        runoff_candidates = [item for item in ballot if (ballot[item] == max_votes)]
    else:
        second_most_votes = votes[1]
        runoff_candidates = [item for item in ballot if (ballot[item] >= second_most_votes)]
    ballot2 = profile.ballot(split, runoff_candidates)
    max_votes = max(ballot2.values())
    winners = [item for item in ballot2 if ballot2[item] == max_votes]
    return winners

@scf
def instant_runoff(profile, tiebreak = 'all', split = True):
    """Until a majority is obtained, eliminate the candidate with the lowest number of votes. Different tiebreak options are possible, but for now, just eliminate all weakest candidates."""
    ballot = profile.ballot(split)
    max_votes = max(ballot.values())
    winners = []
    if (max_votes >= (profile.n / 2.0)):  # majority rule
        for item in ballot:
            if (ballot[item] == max_votes):
                winners.append(item)
        if ((profile.m == 2) or (max_votes > (profile.n / 2.0))):  # either a strict majority winner, or we have two candidates with a 50-50 tie
            return winners
    min_votes = min(ballot.values())
    losers = []
    for item in ballot:
        if (ballot[item] == min_votes):
            losers.append(item)
    new_prof = profile.reduce_to_subset([item for item in profile.universe_list if (item not in losers)])
    return instant_runoff(new_prof, tiebreak = tiebreak, split = split)

@scf
def hare_rule(profile, split = True):
    """Iteratively removes those with least number of first-place votes, until only a winner (or tied winners) remains."""
    ballot = profile.ballot(split)
    min_votes = min(ballot.values())
    if (max(ballot.values()) == min_votes):  # everyone wins
        return profile.universe_list
    remaining_items = [item for item in profile.universe_list if (ballot[item] > min_votes)]
    new_prof = profile.reduce_to_subset(remaining_items)
    return hare_rule(new_prof, split = split)

@scf
def coombs_rule(profile, split = True):
    """Iteratively removes those with the greatest number of last-place votes, until only a winner (or tied winners) remains."""
    rev_profile = profile.reverse()
    loser_ballot = rev_profile.ballot(split)
    max_losing_votes = max(loser_ballot.values())
    if (min(loser_ballot.values()) == max_losing_votes):  # everyone wins
        return profile.universe_list
    remaining_items = [item for item in profile.universe_list if (loser_ballot[item] < max_losing_votes)]
    new_prof = profile.reduce_to_subset(remaining_items)
    return coombs_rule(new_prof, split = split)

@scf
def borda_count(profile):
    """In strict linear ordering, each person contributes a score of (m - k - 1) to the item ranked k (0-up). E.g. if 3 items, gives 2 to first-ranked, 1 to second-ranked, 0 to last-ranked item. Here we modify the rule to allow for indifferent preferences. Order the indifferent ones arbitrarily, but instead of assigning the usual score to each of these, average the scores within the equivalence class."""
    if (not profile.all_prefs_ranked()):
        raise ValueError("All preferences must be complete preorders.")
    borda_counts = dict.fromkeys(profile.universe_list, 0.0)
    for i in range(profile.n):
        if isinstance(profile.prefs[i], StrictRanking):  # convert to a weak ranking
            prefs = [[x] for x in profile.prefs[i].items]
        else:
            prefs = profile.prefs[i].items
        count = profile.m - 1
        for equiv_class in prefs:
            size = len(equiv_class)
            class_score = count - (size - 1) / 2.0
            for item in equiv_class:
                borda_counts[item] += class_score
            count -= size
    max_count = max(borda_counts.values())
    winners = []
    for item in borda_counts:
        if (borda_counts[item] >= max_count):
            winners.append(item)
    return winners

@scf
def blacks_procedure(profile):
    """If a single (weak) Condorcet winner exists, it is the winner. Otherwise, a Borda count determines the outcome."""
    winners = profile.condorcet_winners(False)
    if (len(winners) == 1):
        return winners
    return borda_count(profile)

@scf
def copelands_rule(profile):
    """Does round-robin tournament (each pair is played), and whoever has the most (Wins - Losses) score is the winner."""
    WL = defaultdict(int)
    for x in profile.universe_list:
        for y in profile.universe_list:
            if (x != y):
                if (profile.number_preferring(x, y, True) > profile.number_preferring(y, x, True)):
                    WL[x] += 1
                    WL[y] -= 1
                elif (profile.number_preferring(y, x, True) > profile.number_preferring(x, y, True)):
                    WL[x] -= 1
                    WL[y] += 1
    max_WL = max(WL.values())
    winners = []
    for item in WL:
        if (WL[item] >= max_WL):
            winners.append(item)
    return winners

def minimax(profile, scoring_variant = 1):
    """Winner is the x minimizing the max over y of Score(y, x), where Score(y, x) can be computed in one of three ways:  (let d(x, y) = # strongly preferring x to y)
    1) Winning opposition:  Score(x, y) = d(x, y) if d(x, y) > d(y, x), otherwise 0
    2) Margins:             Score(x, y) = d(x, y) - d(y, x)
    3) Pairwise Opposition: Score(x, y) = d(x, y)"""
    def score(x, y, scoring_variant):
        if (scoring_variant == 1):
            d1 = profile.number_preferring(x, y, True)
            return (d1 if d1 > profile.number_preferring(y, x, True) else 0)
        elif (scoring_variant == 2):
            return (profile.number_preferring(x, y, True) - profile.number_preferring(y, x, True))
        else:
            return profile.number_preferring(x, y, True)
    max_scores = np.zeros(profile.m)
    for i in range(profile.m):
        max_scores[i] = max([score(profile.universe_list[j], profile.universe_list[i], scoring_variant) for j in range(profile.m) if (j != i)])
    minimax = min(max_scores)
    winning_indices = [i for i in range(profile.m) if (max_scores[i] == minimax)]
    winners = [profile.universe_list[i] for i in winning_indices]
    return winners

@scf
def minimax_winning_opp(profile):
    return minimax(profile, scoring_variant = 1)

@scf
def minimax_margin(profile):
    return minimax(profile, scoring_variant = 2)

@scf
def minimax_pairwise_opp(profile):
    return minimax(profile, scoring_variant = 3)
