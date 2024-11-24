#!/usr/bin/env python3
"""Given a suitor Profile, derives a suitee Profile via one of the following schemes:
    reciprocal: suitees rank suitors by decreasing preference for themselves; this makes the suitees as compliant as possible with their earliest proposals in the Gale-Shapley algorithm
    popular: first the suitees are ranked by popularity using rank aggregation; each suitor provided one suitee, so the suitors are thus ranked by their corresponding suitee"""

import argparse
from typing import get_args

import pandas as pd

from pysoc.sct.prefs import Profile
from pysoc.sct.smp import SMPOptions, SuitorRankingMode


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('suitors', help='path to suitor CSV')
    parser.add_argument('-o', '--outfile', help='suitee CSV output path', default='suitees.csv')
    parser.add_argument('--rank-suitors', choices=get_args(SuitorRankingMode), default='reciprocal-popular', help='mode by which suitees will rank suitors')
    parser.add_argument('-a', '--agg', help='rank aggregation method for popularity', choices=('borda', 'kemeny-young'), default='borda')

    args = parser.parse_args()

    print(f'Reading suitor Profile from {args.suitors}')
    df = pd.read_csv(args.suitors)
    suitor_profile = Profile.from_csv(args.suitors, rank_column='Ranked Gifts')
    options = SMPOptions(args.rank_suitors, args.agg)
    suitee_profile = options.get_suitee_profile(suitor_profile, list(df['Brought']))
    print(f'Saving suitee Profile to {args.outfile}')
    suitee_profile.to_csv(args.outfile, two_column=True)
