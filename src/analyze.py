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
from myutils.plot import palettes

PALETTE = palettes['pastel']

##########################################################
def plot_correlations_errbar(corrpath, nepochs, batchsz, outdir):
    """Plot correlation considering the columns paired and unpaired"""
    info(inspect.stack()[0][3] + '()')

    W = 640; H = 480
    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

    df = pd.read_csv(corrpath)
    xs = [ i * batchsz for i in range(nepochs+1)]

    for attrib in ['corrvisits', 'corrfires', 'corrinfec']:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        pp = []
        for paired in np.unique(df.paired):
            marker = 'o' if paired else 'x'
            df2 = df.loc[df.paired == paired]
            for j, top in enumerate(np.unique(df2.top)):
                perepoch = df2.loc[df2.top == top].groupby('epoch')
                ys = perepoch.mean()[attrib]
                yerrs = perepoch.std()[attrib]
                z = ax.errorbar(xs, ys, yerrs, label=top, marker=marker,
                                c=PALETTE[j], alpha=.7)
                pp += z

        ax.set_xlabel('Number of arcs removed')
        ax.set_ylabel('Pearson correlation (degree x ' + attrib + ')')
        ax.set_ylim(0.5, 1.01)
        # l = plt.legend([(pp[0], pp[1])], ['Two keys'], numpoints=1,
               # handler_map={tuple: HandlerTuple(ndivide=None)})
        plt.legend()
        outpath = pjoin(outdir, '{}.png'.format(attrib))
        plt.savefig(outpath); plt.close()

##########################################################
def analyze_sis(betasdir, nepochs, batchsz, outrootdir):
    spldelta = 100 # Sampling delta of this measurement
    for dbeta in sorted(os.listdir(betasdir)):
        if not dbeta.startswith('b'): continue
        allinfs = []
        dbeta2 = pjoin(betasdir, dbeta)
        outdir = pjoin(outrootdir, dbeta)
        os.makedirs(outdir, exist_ok=True)
        for drlz in sorted(os.listdir(dbeta2)):
            drlz2 = pjoin(dbeta2, drlz)
            if not os.path.isdir(drlz2): continue
            infs = np.load(pjoin(drlz2, 'infperepoch.npy'))
            allinfs.append(infs)
        allinfs = np.array(allinfs)
        means = np.mean(allinfs, axis=0)
        stds = np.std(allinfs, axis=0)

        # means = means[:, 1:] # Before transient
        # stds = stds[:, 1:]

        xs = np.arange(means.shape[1]) * spldelta
        for j in range(means.shape[0]):
            # plt.errorbar(xs, means[j, :], stds[j, :])
            plt.plot(xs, means[j, :])
            plt.savefig(pjoin(outdir, '{:02d}.png'.format(j))); plt.close()

##########################################################
def main(outdir):
    info(inspect.stack()[0][3] + '()')

    # corrpath = './data/corrsall.csv'
    # plot_correlations_errbar(corrpath, nepochs=10, batchsz=30, outdir=outdir)

    betasdir = './data/betas/'
    analyze_sis(betasdir, nepochs=10, batchsz=30, outrootdir=outdir)

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
