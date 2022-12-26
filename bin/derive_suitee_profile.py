#!/usr/bin/env python3
"""Given a suitor Profile, derives a suitee Profile via one of the following schemes:
    compliant: suitees rank suitors by decreasing preference for themselves; this makes the suitees as compliant as possible with their earliest proposals in the Gale-Shapley algorithm
    popular: first the suitees are ranked by popularity using rank aggregation; each suitor provided one suitee, so the suitors are thus ranked by their corresponding suitee"""

import argparse

import pandas as pd

from pysoc.sct.prefs import Profile
from pysoc.sct.smp import make_compliant_suitee_profile, make_popular_suitee_profile


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'mode', dest = 'mode')
    compliant = subparsers.add_parser('compliant', help = 'suitees rank suitors by decreasing preference for themselves')
    popular = subparsers.add_parser('popular', help = 'suitees are ranked by rank aggregation; corresponds to ranking on suitors that brought each suitee using "Brought" column of CSV')
    for subparser in [compliant, popular]:
        subparser.add_argument('suitors', help = 'path to suitor CSV')
        subparser.add_argument('-o', '--outfile', help = 'suitee CSV output path', default = 'suitees.csv')
    popular.add_argument('-a', '--agg', help = 'rank aggregation method', choices = ('borda', 'kemeny-young'), default = 'borda')

    args = parser.parse_args()

    print(f'Reading suitor Profile from {args.suitors}')
    suitor_profile = Profile.from_csv(args.suitors, rank_column = 'Ranked Gifts')

    if (args.mode == 'compliant'):
        suitee_profile = make_compliant_suitee_profile(suitor_profile)
    else:  # popular
        df = pd.read_csv(args.suitors, index_col = 0, dtype = str).fillna('')
        suitees = df['Brought']
        suitees_by_suitor = dict(zip(df.index, suitees))
        suitee_profile = make_popular_suitee_profile(suitor_profile, suitees_by_suitor, agg = args.agg)

    print(f'Saving suitee Profile to {args.outfile}')
    suitee_profile.to_csv(args.outfile)