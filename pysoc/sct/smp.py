from collections import defaultdict
import networkx as nx

from pysoc.sct.prefs import Profile, StrictRanking, WeakRanking


def gale_shapley(suitor_prefs, suitee_prefs):
    """Gale-Shapley algorithm to find a stable marriage. Input is a pair of Profiles with StrictRankings, one for the suitors, one for the suitees."""
    assert all(isinstance(ranking, StrictRanking) for ranking in suitor_prefs)
    assert all(isinstance(ranking, StrictRanking) for ranking in suitee_prefs)
    assert (suitor_prefs.universe.issubset(suitee_prefs.names))
    assert (suitee_prefs.universe.issubset(suitor_prefs.names))
    assert (len(suitor_prefs.universe.intersection(suitee_prefs.universe)) == 0), "Suitors and suitees must be unique."
    # get suitors and suitees
    suitors, suitees = suitor_prefs.names, suitee_prefs.names
    num_suitors, num_suitees = len(suitors), len(suitees)
    num_single_suitees = num_suitees
    #num_extra_suitees = max(0, num_suitees - num_suitors)
    #spouses = {name : None for name in suitors + suitees}
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
                num_single_suitees -= 1
        #print(spouse_ranks)
    return (graph, anim_list)

def gale_shapley_weak(suitor_prefs, suitee_prefs):
    """Gale-Shapley algorithm to find a stable marriage. Input is a pair of Profiles with StrictRankings, one for the suitors, one for the suitees."""
    assert all(isinstance(ranking, WeakRanking) for ranking in suitor_prefs)
    assert all(isinstance(ranking, WeakRanking) for ranking in suitee_prefs)
    assert (suitor_prefs.universe.issubset(suitee_prefs.names))
    assert (suitee_prefs.universe.issubset(suitor_prefs.names))
    assert (len(suitor_prefs.universe.intersection(suitee_prefs.universe)) == 0), "Suitors and suitees must be unique."
    # get suitors and suitees
    suitors, suitees = suitor_prefs.names, suitee_prefs.names
    num_suitors, num_suitees = len(suitors), len(suitees)
    diff = num_suitors - num_suitees
    num_single_suitees = num_suitees
    #num_extra_suitees = max(0, num_suitees - num_suitors)
    #spouses = {name : None for name in suitors + suitees}
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
                num_single_suitees -= 1
        #print(spouse_ranks)
    return (graph, anim_list)
