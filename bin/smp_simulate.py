#!/usr/bin/env python3
"""Creates an animation of the Gale-Shapley algorithm given preference data for suitors and suitees."""

import argparse
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pysoc.sct.prefs import CardinalRanking, Profile
from pysoc.sct.sct import SCF_COLLECTION
from pysoc.sct.smp import GaleShapleyAnimator, SMPOptions, SuitorRankingMode, gale_shapley_weak


def read_data(filename):
    # TODO: base path for default image paths
    """Given a filename for a CSV indexed by names, with numeric columns for the rankings (semicolon-delimiting ties), and an optional 'Image' column containing paths to images, returns a Profile containing the weak preferences, and also a dictionary from names to image paths. If no numeric columns are provided, returns None instead of the Profiles."""
    df = pd.read_csv(filename, index_col=0)
    profile = Profile.from_csv(filename, 'Ranked Gifts')
    names = list(df.index)
    # handle image paths
    if 'Image' in df.columns:
        img_paths = {name: path for (name, path) in zip(names, df['Image']) if isinstance(path, str)}
    else:
        img_paths = {}
    return (profile, img_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('suitors', help = 'path to suitor CSV')
    parser.add_argument('suitees', nargs = '?', help = 'path to suitee CSV')
    parser.add_argument('--rank-suitors', choices = get_args(SuitorRankingMode), default = 'reciprocal-popular', help = 'how to rank suitors based on brought gifts when the ranking is not provided')
    parser.add_argument('--random-tiebreak', action = 'store_true', help = 'break ties randomly (instead of interactively)')
    parser.add_argument('-a', '--agg', help = 'rank aggregation method', choices = ('borda', 'kemeny-young'), default = 'borda')
    parser.add_argument('-o', '--outfile', help = 'mp4 output path', default = 'gale_shapley.mp4')
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'verbosity flag')
    parser.add_argument('--figsize', type = int, nargs = 2, default = (12, 8), help = 'figure size in inches (width, height)')
    parser.add_argument('--dpi', type = int, default = 400, help = 'image DPI')
    parser.add_argument('--seed', type = int, help = 'random seed')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Read data
    print(f'Reading suitor data from {args.suitors}...')
    (suitor_prefs, suitor_images) = read_data(args.suitors)

    if args.suitees is None:
        options = SMPOptions(args.rank_suitors, args.agg)
        df = pd.read_csv(args.suitors, index_col=0, dtype=str).fillna('')
        suitee_prefs = options.get_suitee_profile(suitor_prefs, list(df['Brought']))
        suitee_images = {}
    else:
        print(f'Reading suitee data from {args.suitees}...')
        (suitee_prefs, suitee_images) = read_data(args.suitees)

    suitors = suitor_prefs.names
    num_suitors = len(suitors)

    suitee_set = suitor_prefs.universe
    suitees = suitee_prefs.names
    if set(suitees) != suitee_set:
        print("Warning: suitors' target set does not match given suitee set.")
    num_suitees = len(suitees)

    print(f"{num_suitors} suitors: {suitors}")
    print(f"{num_suitors} suitees: {suitees}")

    # Voting (various Social Choice Functions)

    print("\nSuitor preferences (original):\n")
    print(suitor_prefs)

    print("Summary of voting for suitees with various social choice functions:")
    # print("Kemeny-Young: {}".format(kemeny_young(suitor_weak_prefs)))
    print(SCF_COLLECTION.report_all(suitor_prefs))

    print("Suitee preferences (original):\n")
    print(suitee_prefs)

    print("Summary of voting for suitors with various social choice functions:")
    print(SCF_COLLECTION.report_all(suitee_prefs))

    # Gale-Shapley

    print("Suitor preferences:\n")
    print(suitor_prefs)

    print("Suitee preferences:\n")
    print(suitee_prefs)

    print("Running Gale-Shapley...\n")
    (graph, anim_df) = gale_shapley_weak(suitor_prefs, suitee_prefs, verbose=args.verbose, random_tiebreak=args.random_tiebreak)

    animator = GaleShapleyAnimator(suitors, suitees, suitor_images=suitor_images, suitee_images=suitee_images, figsize=args.figsize)
    animator.init_axis()
    animator.plot_nodes()
    node_path = 'tmp.png'
    plt.savefig(node_path, dpi=args.dpi, transparent=True)
    anim = animator.animate(anim_df)

    print(f'Saving movie to {args.outfile}...\n')
    # anim.save(args.outfile, dpi = 400)
    # anim.save(args.outfile, writer = 'ffmpeg', dpi = args.dpi, fps = 30, savefig_kwargs = {'dpi' : args.dpi})
    anim.save(args.outfile, writer='ffmpeg', dpi=args.dpi, fps=30)

    print("\nDONE!\n")
