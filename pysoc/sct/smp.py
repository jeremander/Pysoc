import networkx as nx
from typing import Dict

from pysoc.sct.prefs import CardinalRanking, Profile, StrictRanking
from pysoc.sct.sct import borda_count_ranking, kemeny_young


def gale_shapley(suitor_prefs, suitee_prefs):
    """Gale-Shapley algorithm to find a stable marriage. Input is a pair of Profiles with StrictRankings, one for the suitors, one for the suitees."""
    assert all(isinstance(ranking, StrictRanking) for ranking in suitor_prefs)
    assert all(isinstance(ranking, StrictRanking) for ranking in suitee_prefs)
    assert (suitor_prefs.universe.issubset(suitee_prefs.names))
    assert (suitee_prefs.universe.issubset(suitor_prefs.names))
    assert (len(suitor_prefs.universe.intersection(suitee_prefs.universe)) == 0), "Suitors and suitees must be unique."
    # get suitors and suitees
    suitors, suitees = suitor_prefs.names, suitee_prefs.names
    # initialize the graph
    graph = nx.Graph()
    graph.add_nodes_from(suitors + suitees)
    anim_list = []  # list of pairs (flag, edge), where flag indicates the action to be performed for each edge
    spouse_ranks = {name : -1 for name in suitors + suitees}   # maps from names to spouse ranks (0-up), -1 for unmatched
    matching = {name : None for name in suitors}  # tracks the final matching
    while any(suitee is None for suitee in matching.values()):  # terminate if there are no more unmatched suitors
        for suitor in suitors:  # single suitors propose
            if (graph.degree(suitor) == 0):
                spouse_ranks[suitor] += 1
                suitee = suitor_prefs[suitor][spouse_ranks[suitor]]
                #print("{} proposes to {}".format(suitor, suitee))
                graph.add_edge(suitor, suitee)
                anim_list.append(('add', (suitor, suitee)))
        for suitee in suitees:
            neighbors = set(graph.neighbors(suitee))
            if (len(neighbors) > 1) or ((len(neighbors) > 0) and (spouse_ranks[suitee] < 0)):
                for suitor in suitee_prefs[suitee]:  # find the best-ranked suitor who proposed
                    if (suitor in neighbors):
                        break
                for neighbor in neighbors:  # reject the other neighbors
                    if (neighbor != suitor):
                        #print("{} rejects {}".format(suitee, neighbor))
                        graph.remove_edge(neighbor, suitee)
                        anim_list.append(('reject', (neighbor, suitee)))
                        matching[neighbor] = None
                if (not (matching[suitor] == suitee)):
                    anim_list.append(('keep', (suitor, suitee)))
                    matching[suitor] = suitee
                for neighbor in neighbors:
                    if (neighbor != suitor):
                        anim_list.append(('delete', (neighbor, suitee)))
                spouse_ranks[suitee] = suitee_prefs[suitee].rank(suitor)
                #print("{} is still with {}".format(suitee, suitor))
        #print(spouse_ranks)
    return (graph, anim_list)

def gale_shapley_weak(suitor_prefs, suitee_prefs, verbose = False):
    """Gale-Shapley algorithm to find a stable marriage. Input is a pair of Profiles with weak Rankings, one for the suitors, one for the suitees. Also, the number of suitors and suitees does not have to match."""
    assert all(not isinstance(ranking, StrictRanking) for ranking in suitor_prefs)
    assert all(not isinstance(ranking, StrictRanking) for ranking in suitee_prefs)
    # breakpoint()
    suitee_prefs.names = [str(name) for name in suitee_prefs.names]
    assert (suitor_prefs.universe.issubset(suitee_prefs.names))
    assert (suitee_prefs.universe.issubset(suitor_prefs.names))
    assert (len(suitor_prefs.universe.intersection(suitee_prefs.universe)) == 0), "Suitors and suitees must be unique."
    # convert to strict prefs (breaking ties randomly) to have strict order of proposals
    suitor_strict_prefs = Profile([pref.break_ties_randomly() for pref in suitor_prefs.prefs], names = suitor_prefs.names)
    # get suitors and suitees
    suitors, suitees = suitor_prefs.names, suitee_prefs.names
    # initialize the graph
    graph = nx.Graph()
    graph.add_nodes_from(suitors + suitees)
    anim_list = []  # list of pairs (flag, edge), where flag indicates the action to be performed for each edge
    suitor_ranks = {name : -1 for name in suitors + suitees}   # maps from suitors to match ranks (0-up), -1 for unmatched
    suitor_matching = {name : None for name in suitors}
    suitee_matching = {name : None for name in suitees}
    def unmatched_pair_exists():
        return any(suitee is None for suitee in suitor_matching.values()) and any(suitor is None for suitor in suitee_matching.values())
    while unmatched_pair_exists():  # terminate if there are no more unmatched pairs
        for suitor in suitors:  # single suitors propose
            if (graph.degree(suitor) == 0):
                suitor_ranks[suitor] += 1
                try:
                    suitee = suitor_strict_prefs[suitor][suitor_ranks[suitor]]
                except TypeError:
                    breakpoint()
                if verbose:
                    print("{} proposes to {}".format(suitor, suitee))
                graph.add_edge(suitor, suitee)
                anim_list.append(('add', (suitor, suitee)))
        for suitee in suitees:
            neighbors = set(graph.neighbors(suitee))
            if (len(neighbors) > 1) or ((len(neighbors) > 0) and (suitee_matching[suitee] is None)):
                for suitor_group in suitee_prefs[suitee]:
                    candidates = [suitor for suitor in suitor_group if (suitor in neighbors)]
                    if (len(candidates) == 1):
                        suitor = candidates[0]
                        break
                    elif (len(candidates) > 1):  # break the tie interactively
                        while True:
                            candidate_str = ', '.join(map(str, candidates))
                            suitor = input(f'{suitee} received proposals from: {candidate_str}.\nBreak the tie: ')
                            if (suitor in candidates):
                                break
                            else:
                                print(f"Invalid value '{suitor}'")
                for neighbor in neighbors:  # reject the other neighbors
                    if (neighbor != suitor):
                        if verbose:
                            print(f"{suitee} rejects {neighbor}")
                        graph.remove_edge(neighbor, suitee)
                        anim_list.append(('reject', (neighbor, suitee)))
                        suitor_matching[neighbor] = None
                if (not (suitor_matching[suitor] == suitee)):
                    anim_list.append(('keep', (suitor, suitee)))
                    suitor_matching[suitor] = suitee
                    suitee_matching[suitee] = suitor
                for neighbor in neighbors:
                    if (neighbor != suitor):
                        anim_list.append(('delete', (neighbor, suitee)))
                if verbose:
                    print(f"{suitee} is still with {suitor}")
    return (graph, anim_list)

def make_compliant_suitee_profile(suitor_profile: Profile) -> Profile:
    """Given a suitor Profile, creates a suitee Profile where they rank suitors by decreasing preference for themselves. This makes the suitees as compliant as possible with their earliest proposals in the Gale-Shapley algorithm."""
    suitors = suitor_profile.names
    suitees = sorted(list(suitor_profile.universe))
    suitee_rankings = []
    for suitee in suitees:
        suitee_rankings.append(CardinalRanking({suitor : -suitor_profile[suitor].rank(suitee) for suitor in suitors}))
    return Profile(suitee_rankings, names = suitees)

def make_popular_suitee_profile(suitor_profile: Profile, suitees_by_suitor: Dict[str, str], agg: str = 'borda') -> Profile:
    """First rank the suitees by popularity using rank aggregation then rank the suitors by their corresponding suitee (each suitor "provided" a suitee)."""
    if (agg == 'borda'):
        suitor_ranking = borda_count_ranking(suitor_profile)
    elif (agg == 'kemeny-young'):
        suitor_ranking = kemeny_young(suitor_profile)
    else:
        raise ValueError(f"Unknown aggregation method: {agg!r}")
    suitors_by_suitee = {}
    for (suitor, suitee) in suitees_by_suitor.items():
        suitors_by_suitee[suitee] = suitor
    suitors = suitor_profile.names
    suitee_ranking = CardinalRanking({suitor : suitor_ranking.score_dict[suitees_by_suitor[suitor]] for suitor in suitors})
    suitees = sorted(list(suitor_profile.universe))
    suitee_rankings = []
    for suitee in suitees:
        suitee_rankings.append(suitee_ranking)
    return Profile(suitee_rankings, names = suitees)
