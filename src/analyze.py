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
def plot_correlations_errbar(df, attrib, label, ax=None):
    info(inspect.stack()[0][3] + '()')

    nepochs = 10
    batchsz = 30
    x = [ i * batchsz for i in range(nepochs+1)]

    col = 'corr' + attrib # Expects the naming fvisitis or ffires
    for top in np.unique(df.top):
        y = df.loc[df.top == top].groupby('epoch').mean()[col]
        yerr = df.loc[df.top == top].groupby('epoch').std()[col]
        # y = df.loc[df.top == top][col]
        ax.errorbar(x, y, yerr, label='[{}] {}]'.format(label, top))

    ax.set_xlabel('Number of arcs removed')
    ax.set_ylabel('Pearson correlation (degree x ' + attrib + ')')
    ax.set_ylim(0.5, 1.01)
    plt.legend()
    return fig, ax

##########################################################
def plot_correlations_all(corrpath, outdir):
    """Plot correlation considering the columns paired and unpaired"""
    info(inspect.stack()[0][3] + '()')

    df = pd.read_csv(corrpath)
    W = 640; H = 480

    for col in ['visits', 'fires', 'infecs']:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        fig, ax = plot_correlations_errbar(df.loc[df.paired==False], col, 'unpaired',
                                           ax, outdir)
        plot_correlations_errbar(df.loc[df.paired==True], col, 'paired', ax, outdir)
        outpath = pjoin(outdir, '{}.png'.format(col))
        plt.savefig(outpath); plt.close()

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    corrpath = './data/corrsall.csv'
    plot_correlations_all(corrpath, outdir)

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
