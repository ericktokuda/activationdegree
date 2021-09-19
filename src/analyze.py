#!/usr/bin/env python3
"""Analyze the results from run.py
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from myutils import info, create_readme

##########################################################
def plot_correlations_errbar(df, attrib, outdir):
    info(inspect.stack()[0][3] + '()')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    nepochs = 10
    batchsz = 30
    x = [ i * batchsz for i in range(nepochs+1)]

    col = 'f' + attrib # Expects the naming fvisitis or ffires
    for top in np.unique(df.top):
        y = df.loc[df.top == top].groupby('epoch').mean()[col]
        yerr = df.loc[df.top == top].groupby('epoch').std()[col]
        # y = df.loc[df.top == top][col]
        ax.errorbar(x, y, yerr, label=top)

    ax.set_xlabel('Number of arcs removed')
    ax.set_ylabel('Number of ' + attrib)
    plt.legend()
    outpath = pjoin(outdir, '{}.png'.format(col))
    plt.savefig(outpath)

##########################################################
def plot_correlations_all(corrpath, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    df = pd.read_csv(corrpath)
    plot_correlations_errbar(df, 'visits', outdir)
    plot_correlations_errbar(df, 'fires', outdir)

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    corrpath = 'data/corrsall.csv'
    plot_correlations_all(corrpath, outdir)

    info('For Aiur!')


##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
