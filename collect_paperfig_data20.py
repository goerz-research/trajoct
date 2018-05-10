#!/usr/bin/env python
"""Collect the data for the figures that will be shown in the paper"""

import os
from os.path import join
from glob import glob
from textwrap import dedent
from collections import OrderedDict
import numpy as np
import pandas as pd

from src.notebook_plots_v1 import (
    collect_noise_table, combine_noise_tables)
from src.pulse_smoothing_v1 import smooth_pulse_savgol


def rf_sort_key(rf):
    """Key for sorting runfolder according to number of trajectories"""
    try:
        a, b = rf.split('_ntrajs')
        return (a, int(b))
    except ValueError:
        return (rf, 0)


def write_convergence_data(root, outfile):
    """write a csv file for the table of convergence data"""
    rfs = OrderedDict([
        ('independent', sorted(
            glob(join(root, '20nodesT50_independent_ntrajs*')), key=rf_sort_key)),
        ('rho', [join(root, '20nodesT50_rho_ntrajs1')]),
        ('cross', sorted(
            glob(join(root, '20nodesT50_cross_ntrajs*')), key=rf_sort_key)),
    ])

    data = OrderedDict()
    for category in rfs.keys():
        for rf in rfs[category]:
            n_trajs = int(rf.split("_ntrajs")[-1])
            col_label = (category, n_trajs)  # MultiIndex tuple
            datfile = join(rf, 'rho_prop_error.dat')
            try:
                oct_iter = np.genfromtxt(
                    datfile,  dtype=int, usecols=0, unpack=True)
                err = np.genfromtxt(
                    datfile,  dtype=float, usecols=1, unpack=True)
                data[col_label] = pd.Series(err, index=oct_iter)
            except OSError as exc_info:
                print("Exception: %r" % exc_info)
                continue
    df = pd.DataFrame(data=data)
    df.columns = df.columns.rename(('category', 'n_trajs'))
    df.index = df.index.rename('iter')
    df.loc[:, ('cross', 2)] = df.loc[:, ('cross', 2)].fillna(
        value=df.loc[:, ('cross', 8)])
    df.loc[:, ('cross', 4)] = df.loc[:, ('cross', 4)].fillna(
        value=df.loc[:, ('cross', 8)])

    with open(outfile, "w") as out_fh:
        out_fh.write(dedent(r'''
        # convergence data
        # read data in Python with:
        #     import pandas as pd
        #     df = pd.read_csv(open('{outfile}'), index_col=0, header=[4,5])
        ''').lstrip().format(outfile=outfile))
        out_fh.write(df.to_csv())
    print("Written %s" % outfile)


if __name__ == "__main__":
    outfolder = './paperfig_data20'
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    write_convergence_data(
        root='./data/oct_single_excitation',
        outfile=join(outfolder, 'convergence20.csv'))
