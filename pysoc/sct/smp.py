from collections import Counter
import math
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from typing import Dict, NamedTuple, Tuple

from pysoc.sct.prefs import CardinalRanking, Profile, Ranking, StrictRanking
from pysoc.sct.sct import borda_count_ranking, kemeny_young

# raise size limit for animations
matplotlib.rcParams['animation.embed_limit'] = 2 ** 31


def gale_shapley(suitor_prefs: Profile, suitee_prefs: Profile) -> Tuple[nx.Graph, pd.DataFrame]:
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
    actions = []  # list of (action, suitor, suitee)
    spouse_ranks = {name : -1 for name in suitors + suitees}   # maps from names to spouse ranks (0-up), -1 for unmatched
    matching = {name : None for name in suitors}  # tracks the final matching
    while any(suitee is None for suitee in matching.values()):  # terminate if there are no more unmatched suitors
        for suitor in suitors:  # single suitors propose
            if (graph.degree(suitor) == 0):
                spouse_ranks[suitor] += 1
                suitee = suitor_prefs[suitor][spouse_ranks[suitor]]
                #print("{} proposes to {}".format(suitor, suitee))
                graph.add_edge(suitor, suitee)
                actions.append(('add', suitor, suitee))
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
                        actions.append(('reject', neighbor, suitee))
                        matching[neighbor] = None
                if (not (matching[suitor] == suitee)):
                    actions.append(('keep', suitor, suitee))
                    matching[suitor] = suitee
                for neighbor in neighbors:
                    if (neighbor != suitor):
                        actions.append(('delete', neighbor, suitee))
                spouse_ranks[suitee] = suitee_prefs[suitee].rank(suitor)
                #print("{} is still with {}".format(suitee, suitor))
        #print(spouse_ranks)
    actions_df = pd.DataFrame(actions, columns = ['action', 'suitor', 'suitee'])
    return (graph, actions_df)

def gale_shapley_weak(suitor_prefs: Profile, suitee_prefs: Profile, verbose: bool = False, random_tiebreak: bool = False) -> Tuple[nx.Graph, pd.DataFrame]:
    """Gale-Shapley algorithm to find a stable marriage. Input is a pair of Profiles with weak Rankings, one for the suitors, one for the suitees. Also, the number of suitors and suitees does not have to match."""
    assert all(not isinstance(ranking, StrictRanking) for ranking in suitor_prefs)
    assert all(not isinstance(ranking, StrictRanking) for ranking in suitee_prefs)
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
    actions = []  # list of (action, suitor, suitee) tuples
    suitor_ranks = {name : -1 for name in suitors + suitees}   # maps from suitors to match ranks (0-up), -1 for unmatched
    suitor_matching = {name : None for name in suitors}
    suitee_matching = {name : None for name in suitees}
    def unmatched_pair_exists():
        return any(suitee is None for suitee in suitor_matching.values()) and any(suitor is None for suitor in suitee_matching.values())
    while unmatched_pair_exists():  # terminate if there are no more unmatched pairs
        for suitor in suitors:  # single suitors propose
            if (graph.degree(suitor) == 0):
                suitor_ranks[suitor] += 1
                suitee = suitor_strict_prefs[suitor][suitor_ranks[suitor]]
                if verbose:
                    print(f'{suitor} proposes to {suitee}')
                graph.add_edge(suitor, suitee)
                actions.append(('add', suitor, suitee))
        for suitee in suitees:
            neighbors = set(graph.neighbors(suitee))
            if (len(neighbors) > 1) or ((len(neighbors) > 0) and (suitee_matching[suitee] is None)):
                for suitor_group in suitee_prefs[suitee]:
                    candidates = [suitor for suitor in suitor_group if (suitor in neighbors)]
                    if (len(candidates) == 1):
                        suitor = candidates[0]
                        break
                    elif (len(candidates) > 1):
                        if random_tiebreak:  # break the tie randomly
                            suitor = np.random.choice(candidates)
                        else:  # break the tie interactively
                            while True:
                                candidate_str = ', '.join(map(str, candidates))
                                suitor = input(f'{suitee} received proposals from: {candidate_str}.\nBreak the tie: ')
                                if (suitor in candidates):
                                    break
                                else:
                                    print(f"Invalid value '{suitor}'")
                        break
                for neighbor in neighbors:  # reject the other neighbors
                    if (neighbor != suitor):
                        if verbose:
                            print(f"{suitee} rejects {neighbor}")
                        graph.remove_edge(neighbor, suitee)
                        actions.append(('reject', neighbor, suitee))
                        suitor_matching[neighbor] = None
                if (not (suitor_matching[suitor] == suitee)):
                    actions.append(('keep', suitor, suitee))
                    suitor_matching[suitor] = suitee
                    suitee_matching[suitee] = suitor
                for neighbor in neighbors:
                    if (neighbor != suitor):
                        actions.append(('delete', neighbor, suitee))
                if verbose:
                    print(f"{suitee} is still with {suitor}")
    action_df = pd.DataFrame(actions, columns = ['action', 'suitor', 'suitee'])
    return (graph, action_df)

def num_weak_compositions(n: int, k: int) -> int:
    return 1 if (n == k == 0) else math.comb(n + k - 1, n)

def num_worse_signatures(n: int, signature: Tuple[int, ...]) -> int:
    if (n == 0):
        return 0
    k = len(signature)
    i0 = signature[0]
    num_sub_signatures = sum(num_weak_compositions(n - i, k - 1) for i in range(i0))
    return num_sub_signatures + num_worse_signatures(n - i0, signature[1:])

class RankSignature(NamedTuple):
    n: int  # number of raters
    signature: Tuple[int, ...]  # number of instances of each rank
    @property
    def happiness_score(self) -> float:
        """Computes percentile of the signature out of all possible signatures, resulting in a score from 0 to 100."""
        num_sigs = num_weak_compositions(self.n, self.n)
        num_worse_sigs = num_worse_signatures(self.n, self.signature)
        return 0.0 if (num_worse_sigs == 0) else (100.0 * (num_worse_sigs / (num_sigs - 1)))

def get_rank_signatures(suitor_profile: Profile, suitee_profile: Profile, graph: nx.Graph) -> Tuple[RankSignature, RankSignature]:
    """Gets the rank signature for suitors and suitees, given a matching."""
    suitor_counts, suitee_counts = Counter(), Counter()
    num_suitors, num_suitees = len(suitor_profile), len(suitee_profile)
    for (ranking, suitor) in zip(suitor_profile, suitor_profile.names):
        suitee = next(iter(graph.neighbors(suitor)))
        suitor_counts[ranking.rank(suitee)] += 1
    suitor_sig = RankSignature(num_suitors, tuple(suitor_counts[i] for i in range(num_suitors)))
    for (ranking, suitee) in zip(suitee_profile, suitee_profile.names):
        suitor = next(iter(graph.neighbors(suitee)))
        suitee_counts[ranking.rank(suitor)] += 1
    suitee_sig = RankSignature(num_suitees, tuple(suitee_counts[i] for i in range(num_suitees)))
    return (suitor_sig, suitee_sig)

def make_reciprocal_suitee_profile(suitor_profile: Profile) -> Profile:
    """Given a suitor Profile, creates a suitee Profile where they rank suitors by decreasing preference for themselves. This makes the suitees as compliant as possible with their earliest proposals in the Gale-Shapley algorithm."""
    suitors = suitor_profile.names
    suitees = sorted(list(suitor_profile.universe))
    suitee_rankings = []
    for suitee in suitees:
        suitee_rankings.append(CardinalRanking({suitor : -suitor_profile[suitor].rank(suitee) for suitor in suitors}))
    return Profile(suitee_rankings, names = suitees)

def aggregate_ranking(profile: Profile, agg: str = 'borda') -> Ranking:
    """Given a profile and rank aggregation method, returns a Ranking which aggregates the profile."""
    if (agg == 'borda'):
        return borda_count_ranking(profile)
    elif (agg == 'kemeny-young'):
        # TODO: make this a weak ranking?
        return kemeny_young(profile)[0]
    else:
        raise ValueError(f'Unknown aggregation method: {agg!r}')

def make_popular_suitee_profile(suitor_profile: Profile, suitees_by_suitor: Dict[str, str], agg: str = 'borda') -> Profile:
    """First rank the suitees by popularity using rank aggregation then rank the suitors by their corresponding suitee (each suitor "provided" a suitee)."""
    suitor_ranking = aggregate_ranking(suitor_profile, agg = agg)
    suitors_by_suitee = {}
    for (suitor, suitee) in suitees_by_suitor.items():
        suitors_by_suitee[suitee] = suitor
    suitors = suitor_profile.names
    suitee_ranking = CardinalRanking({suitor : suitor_ranking.score_dict[suitees_by_suitor[suitor]] for suitor in suitors})
    suitees = sorted(suitor_profile.universe)
    suitee_rankings = []
    for _ in suitees:
        suitee_rankings.append(suitee_ranking)
    return Profile(suitee_rankings, names = suitees)

def make_border(img: Image, border_frac: float = 0.10) -> Image:
    size = img.size
    c = 1 + 2 * border_frac
    new_size = (int(c * size[0]), int(c * size[1]))
    border = min((new_size[0] - size[0]) // 2, (new_size[1] - size[1]) // 2)
    return ImageOps.expand(img, border, fill = 'white')

def make_thumbnail(img: Image, width: int = 200) -> Image:
    size = img.size
    ratio = width / size[0]
    return img.resize((int(size[0] * ratio), int(size[1] * ratio)), Image.ANTIALIAS)

class GaleShapleyAnimator:
    def __init__(self, suitors, suitees, suitor_images = dict(), suitee_images = dict(), figsize = None, thumbnail_width: int = 200):
        self.suitors = suitors
        self.suitees = suitees
        self.suitor_images = suitor_images
        self.suitee_images = suitee_images
        self.figsize = figsize
        self.thumbnail_width = thumbnail_width
        self.nodes = suitors + suitees
        num_suitors, num_suitees = len(suitors), len(suitees)
        self.N = max(num_suitors, num_suitees)
        self.height = max(2, self.N // 2)
        self.pos = {node : (i, self.height) if (i < num_suitors) else (i - num_suitors, 0) for (i, node) in enumerate(self.nodes)}
        self.node_colors = {node : '#3BB9FF' if (i < num_suitors) else '#F778A1' for (i, node) in enumerate(self.nodes)}
    def init_axis(self):
        (self.fig, self.ax) = plt.subplots(figsize = self.figsize)
        plt.subplots_adjust(left = 0.0, right = 1.0, top = 1.0, bottom = 0.0)
        # self.fig.tight_layout()
        self.ax.set_xlim((-0.5, self.N - 0.5))
        self.ax.set_ylim((-0.5, self.height + 0.5))
        self.ax.axis('off')
        self.ax.set_aspect('equal')
    def plot_nodes(self):
        def gen_nodes():
            for suitor in self.suitors:
                yield (suitor, self.suitor_images.get(suitor))
            for suitee in self.suitees:
                yield (suitee, self.suitee_images.get(suitee))
        for (node, img_file) in gen_nodes():
            p = self.pos[node]
            if img_file:  # use image
                img = Image.open(img_file)
                # img = make_border(img, 0.8)
                img = make_thumbnail(img, width = self.thumbnail_width)
                self.ax.imshow(img, origin = 'upper', extent = [p[0] - 0.4, p[0] + 0.4, p[1] - 0.4, p[1] + 0.4])
            else:  # use the name
                circ = plt.Circle(p, 0.35, color = self.node_colors[node])
                self.ax.add_artist(circ)
                self.ax.text(*p, node, ha = 'center', va = 'center', fontweight = 'bold', fontsize = 12)
    def animate(self, anim_df, squash: float = 0.8):
        """Animates the Gale-Shapley algorithm. Takes list of suitors, suitees, and animation actions returned by the gale_shapley function."""
        anim_list = [(action, (suitor, suitee)) for (_, action, suitor, suitee) in anim_df.itertuples()]
        self.init_axis()
        lines = []
        def init():
            self.plot_nodes()
            return self.ax.artists
        def update(frame):
            print(f'frame = {frame}')
            if (frame > 0):
                nonlocal lines
                (flag, edge) = anim_list[frame - 1]
                xdata, ydata = zip(self.pos[edge[0]], self.pos[edge[1]])
                # squash the lines by the given factor
                top = (1 + squash) / 2
                bottom = 1 - top
                ydata = (top * self.height, bottom * self.height)
                def get_line_index():
                    for (i, line) in enumerate(lines):
                        if (tuple(line.get_xdata()) == xdata) and (tuple(line.get_ydata()) == ydata):
                            return i
                if (flag == 'add'):
                    lines += self.ax.plot(xdata, ydata, linewidth = 2, color = 'black', linestyle = 'dashed')
                elif (flag == 'delete'):
                    i = get_line_index()
                    l = lines.pop(i)
                    l.remove()
                    del(l)
                elif (flag == 'reject'):
                    i = get_line_index()
                    lines[i].set_color('red')
                    lines[i].set_linestyle('dashed')
                elif (flag == 'keep'):
                    i = get_line_index()
                    lines[i].set_color('green')
                    lines[i].set_linestyle('solid')
                else:
                    raise ValueError("Invalid flag: '{}'".format(flag))
            return lines
        anim = animation.FuncAnimation(self.fig, update, init_func = init, frames = len(anim_list) + 1, interval = 1000, blit = True)
        return anim