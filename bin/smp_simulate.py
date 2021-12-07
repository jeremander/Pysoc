#!/usr/bin/env python3
"""Creates an animation of the Gale-Shapley algorithm given preference data for suitors and suitees."""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from pysoc.sct.prefs import CardinalRanking, Profile
from pysoc.sct.sct import SCF_COLLECTION, kemeny_young
from pysoc.sct.smp import gale_shapley_weak, make_compliant_suitee_profile


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
    """Given a filename for a CSV indexed by names, with numeric columns for the rankings (semicolon-delimiting ties), and an optional 'Image' column containing paths to images, returns a Profile containing the weak preferences, and also a dictionary from names to image paths. If no numeric columns are provided, returns None instead of the Profiles."""
    profile = Profile.from_csv(filename)
    df = pd.read_csv(filename, index_col = 0)
    names = list(df.index)
    # handle image paths
    if ('Image' in df.columns):
        img_path_dict = dict((name, path) for (name, path) in zip(names, df['Image']) if isinstance(path, str))
    else:
        img_path_dict = dict()
    return (profile, img_path_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('suitors', help = 'path to suitor CSV')
    parser.add_argument('suitees', help = 'path to suitee CSV')
    parser.add_argument('-o', '--outfile', help = 'mp4 output path', default = 'gale_shapley.mp4')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'verbosity flag')
    args = parser.parse_args()

    # Read data
    print("Reading suitor data from {}...".format(args.suitors))
    (suitor_prefs, suitor_img_path_dict) = read_data(args.suitors)

    if (args.suitees is None):
        suitee_prefs = make_compliant_suitee_profile(suitor_prefs)
        suitee_img_path_dict = dict()
    else:
        print("Reading suitee data from {}...".format(args.suitees))
        (suitee_prefs, suitee_img_path_dict) = read_data(args.suitees)

    suitors = suitor_prefs.names
    num_suitors = len(suitors)

    suitee_set = suitor_prefs.universe
    if (suitee_prefs is None):  # rank the suitors based on suitors' own preferences
        suitees = sorted(list(suitee_set))
        suitee_rankings = []
        for suitee in suitees:
            suitee_rankings.append(CardinalRanking({suitor : -suitor_prefs[suitor].rank(suitee) for suitor in suitors}))
        suitee_prefs = Profile(suitee_rankings, names = suitees)
    else:
        suitees = suitee_prefs.names
        if (set(suitees) != suitee_set):
            print("Warning: suitors' target set does not match given suitee set.")
    num_suitees = len(suitees)

    print(f"{num_suitors} suitors: {suitors}")
    print(f"{num_suitors} suitees: {suitees}")

    # Voting (various Social Choice Functions)

    print("\nSuitor preferences (original):\n")
    print(suitor_prefs)

    print("Summary of voting for suitees with various social choice functions:")
    #print("Kemeny-Young: {}".format(kemeny_young(suitor_weak_prefs)))
    print(SCF_COLLECTION.report_all(suitor_prefs))

    print("Suitee preferences (original):\n")
    print(suitee_prefs)

    # TODO: broken for total indifference
    #print("Summary of voting for suitors with various social choice functions:")
    #print(SCF_COLLECTION.report_all(suitee_prefs))

    # Gale-Shapley

    print("Suitor preferences:\n")
    print(suitor_prefs)

    print("Suitee preferences:\n")
    print(suitee_prefs)

    print("Running Gale-Shapley...\n")
    (graph, anim_list) = gale_shapley_weak(suitor_prefs, suitee_prefs, verbose = args.verbose)

    img_path_dict = suitor_img_path_dict
    img_path_dict.update(suitee_img_path_dict)

    anim = gs_animate(suitors, suitees, anim_list, img_path_dict = img_path_dict)

    print("Saving movie to {}...\n".format(args.outfile))
    anim.save(args.outfile, dpi = 400)

    print("\nDONE!\n")
