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
            glob(join(root, 'T5_independent_ntrajs*')), key=rf_sort_key)),
        ('rho', [join(root, 'T5_rho_ntrajs1')]),
        ('cross', sorted(
            glob(join(root, 'T5_cross_ntrajs*')), key=rf_sort_key)),
    ])

    data = OrderedDict()
    for category in rfs.keys():
        for rf in rfs[category]:
            n_trajs = int(rf.split("_ntrajs")[-1])
            col_label = (category, n_trajs)  # MultiIndex tuple
            datfile = join(rf, 'rho_prop_error.dat')
            oct_iter = np.genfromtxt(
                datfile,  dtype=int, usecols=0, unpack=True)
            err = np.genfromtxt(
                datfile,  dtype=float, usecols=1, unpack=True)
            data[col_label] = pd.Series(err, index=oct_iter)
    df = pd.DataFrame(data=data)
    df.columns = df.columns.rename(('category', 'n_trajs'))
    df.index = df.index.rename('iter')

    with open(outfile, "w") as out_fh:
        out_fh.write(dedent(r'''
        # convergence data
        # read data in Python with:
        #     import pandas as pd
        #     df = pd.read_csv(open('{outfile}'), index_col=0, header=[4,5])
        ''').lstrip().format(outfile=outfile))
        out_fh.write(df.to_csv())
    print("Written %s" % outfile)


def write_noise_data(root, outfile):
    """write a csv file for the table of noise data"""
    rfs_independent_trajs = sorted(
        glob(join(root, 'T5_independent_ntrajs*')), key=rf_sort_key)
    rfs_cross_trajs = sorted(
        glob(join(root, 'T5_cross_ntrajs*')), key=rf_sort_key)
    df = combine_noise_tables([
        collect_noise_table(
            rfs_independent_trajs, smooth_pulse_savgol,
            window_length=5, polyorder=3),
        collect_noise_table(
            rfs_cross_trajs, smooth_pulse_savgol,
            window_length=5, polyorder=3)],
        ['independent', 'cross'])
    df.index = df.index.set_names('ntrajs')
    with open(outfile, "w") as out_fh:
        out_fh.write(dedent(r'''
        # noise data
        # read data in Python with:
        #     import pandas as pd
        #     df = pd.read_csv('{outfile}', index_col=0, header=4)
        ''').lstrip().format(outfile=outfile))
        out_fh.write(df.to_csv())
    print("Written %s" % outfile)


if __name__ == "__main__":
    outfolder = './paperfig_data'
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    write_convergence_data(
        root='./data/method_comparison_dicke2',
        outfile=join(outfolder, 'convergence.csv'))
    write_noise_data(
        root='./data/method_comparison_dicke2_noise',
        outfile=join(outfolder, 'noise.csv'))
