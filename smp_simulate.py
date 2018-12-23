#!/usr/bin/env python3
"""Creates an animation of the Gale-Shapley algorithm given preference data for suitors and suitees."""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import sys

from pysoc.sct.prefs import Profile, Ranking, StrictRanking
from pysoc.sct.sct import SCF_COLLECTION, kemeny_young
from pysoc.sct.smp import gale_shapley


def gs_animate(suitors, suitees, anim_list, img_path_dict = dict()):
    """Animates the Gale-Shapley algorithm. Takes list of suitors, suitees, and animation actions returned by the gale_shapley function. img_path_dict is a dictionary from suitor/suitee names to image paths."""
    nodes = suitors + suitees
    num_suitors, num_suitees = len(suitors), len(suitees)
    N = max(num_suitors, num_suitees)
    height = N // 2
    pos = {node : (i, height) if (i < num_suitors) else (i - num_suitors, 0) for (i, node) in enumerate(nodes)}
    node_colors = {node : '#3BB9FF' if (i < num_suitors) else '#F778A1' for (i, node) in enumerate(nodes)}
    (fig, ax) = plt.subplots()
    lines = []
    def init():
        ax.set_xlim((-0.5, N - 0.5))
        ax.set_ylim((-0.5, height + 0.5))
        ax.axis('off')
        fig.tight_layout()
        #fig.set_size_inches(16, 8)
        ax.set_aspect('equal')
        return plot_nodes()
    def plot_nodes():
        artists = []
        for (node, p) in pos.items():
            if (node in img_path_dict):  # use image
                img = Image.open(img_path_dict[node])
                artists.append(ax.imshow(img, origin = 'upper', extent = [p[0] - 0.4, p[0] + 0.4, p[1] - 0.4, p[1] + 0.4]))
            else:  # use the name
                #artists.append(ax.scatter([p[0]], [p[1]], s = 1000, color = node_colors[node]))
                circ = plt.Circle(p, 0.35, color = node_colors[node])
                ax.add_artist(circ)
                artists.append(circ)
                artists.append(ax.text(*p, node, ha = 'center', va = 'center', fontweight = 'bold', fontsize = 12))
        return artists
    def update(frame):
        print("frame = {}".format(frame))
        if (frame > 0):
            nonlocal lines
            (flag, edge) = anim_list[frame - 1]
            xdata, ydata = zip(pos[edge[0]], pos[edge[1]])
            ydata = (0.9 * height, 0.1 * height)  # squash the lines a little
            def get_line_index():
                for (i, line) in enumerate(lines):
                    if (tuple(line.get_xdata()) == xdata) and (tuple(line.get_ydata()) == ydata):
                        return i
            if (flag == 'add'):
                lines += ax.plot(xdata, ydata, linewidth = 2, color = 'black', linestyle = 'dashed')
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
    anim = animation.FuncAnimation(fig, update, init_func = init, frames = len(anim_list) + 1, interval = 1000, blit = True)
    return anim

def read_data(filename):
    """Given a filename for a CSV indexed by names, with numeric columns for the rankings (semicolon-delimiting ties), and an optional 'Image' column containing paths to images, returns two Profiles, one for weak and one for strong preferences, and a dictionary from names to image paths. If no numeric columns are provided, returns None instead of the Profiles."""
    df = pd.read_csv(filename, index_col = 0)
    numeric_cols = sorted([col for col in df.columns if all(c.isdigit() for c in col)], key = lambda x : int(x))
    names = list(df.index)
    # handle Profiles
    if (len(numeric_cols) == 0):
        weak_profile = None
        strict_profile = None
    else:
        weak_rankings = []
        strict_rankings = []
        for (_, row) in df[numeric_cols].iterrows():
            weak_ranking, strict_ranking = [], []
            for x in row:
                if isinstance(x, str):
                    tie = x.split(';')
                    weak_ranking.append(tie)
                    strict_ranking += [tie[i] for i in np.random.permutation(len(tie))]  # break ties randomly
            weak_rankings.append(weak_ranking)
            strict_rankings.append(strict_ranking)
        target_set = set(strict_rankings[0])
        try:
            assert all(set(strict_ranking) == target_set for strict_ranking in strict_rankings)
        except AssertionError:
            for strict_ranking in strict_rankings:
                if (set(strict_ranking) != target_set):
                    print(strict_ranking)
            raise AssertionError
        weak_profile = Profile([Ranking(weak_ranking) for weak_ranking in weak_rankings], names = names)
        strict_profile = Profile([StrictRanking(strict_ranking) for strict_ranking in strict_rankings], names = names)
    # handle image paths
    if ('Image' in df.columns):
        img_path_dict = dict((name, path) for (name, path) in zip(names, df['Image']) if isinstance(path, str))
    else:
        img_path_dict = dict()
    return (weak_profile, strict_profile, img_path_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('suitors', nargs = 1, help = 'path to suitor CSV', type = str)
    parser.add_argument('suitees', nargs = 1, help = 'path to suitee CSV', default = None, type = str)
    parser.add_argument('-o', '--outfile', help = 'mp4 output path', default = 'gale_shapley.mp4', type = str)
    args = parser.parse_args()

    # Read data

    suitor_filename = args.suitors[0]
    suitee_filename = args.suitees if (args.suitees is None) else args.suitees[0]

    print("\nReading suitor data from {}...\n".format(suitor_filename))
    (suitor_weak_prefs, suitor_strict_prefs, suitor_img_path_dict) = read_data(suitor_filename)

    if (suitee_filename is None):
        (suitee_weak_prefs, suitee_strict_prefs, suitee_img_path_dict) = (None, None, dict())
    else:
        print("\nReading suitee data from {}...\n".format(suitee_filename))
        (suitee_weak_prefs, suitee_strict_prefs, suitee_img_path_dict) = read_data(suitee_filename)

    suitors = suitor_weak_prefs.names
    num_suitors = len(suitors)

    suitee_set = set(suitor_strict_prefs[suitors[0]].items)  # infer suitors from suitee prefs
    if (suitee_weak_prefs is None):  # randomly generate the suitee prefs
        print("Randomly generating suitee preferences...\n")
        suitees = sorted(list(suitee_set))
        suitee_weak_prefs = suitee_strict_prefs = Profile.random(suitors, names = suitees)
    else:
        suitees = suitee_weak_prefs.names
        if (set(suitees) != suitee_set):
            print("Warning: suitors' target set does not match given suitee set.")
    num_suitees = len(suitees)

    print("{} suitors: [{}]".format(num_suitors, ', '.join(map(str, suitors))))
    print("{} suitees: [{}]".format(num_suitees, ', '.join(map(str, suitees))))

    # Voting (various Social Choice Functions)

    print("\nSuitor preferences (original):\n")
    print(suitor_weak_prefs)

    print("Summary of voting for suitees with various social choice functions:")
    #print("Kemeny-Young: {}".format(kemeny_young(suitor_weak_prefs)))
    print(SCF_COLLECTION.report_all(suitor_weak_prefs))

    print("Suitee preferences (original):\n")
    print(suitee_weak_prefs)

    print("Summary of voting for suitees with various social choice functions:")
    print(SCF_COLLECTION.report_all(suitee_weak_prefs))    

    # Gale-Shapley 

    print("Suitor preferences (tiebroken):\n")
    print(suitor_strict_prefs)

    print("Suitee preferences (tiebroken):\n")
    print(suitee_strict_prefs)

    print("Running Gale-Shapley...\n")
    (graph, anim_list) = gale_shapley(suitor_strict_prefs, suitee_strict_prefs)


    img_path_dict = suitor_img_path_dict
    img_path_dict.update(suitee_img_path_dict)

    anim = gs_animate(suitors, suitees, anim_list, img_path_dict = img_path_dict)

    print("Saving movie to {}...\n".format(args.outfile))
    anim.save(args.outfile, dpi = 400)

    print("\nDONE!\n")
