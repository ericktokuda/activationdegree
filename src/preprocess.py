#!/usr/bin/env python3
"""Preprocessing prior to run simulation
"""

import argparse
import time, datetime
import os, random
from os.path import join as pjoin
import inspect
from types import SimpleNamespace

import sys
import shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import igraph
from scipy.stats import pearsonr
from myutils import info, create_readme
import json
import scipy
import scipy.sparse as spa
from multiprocessing import Pool
import time
import pandas as pd

##########################################################
SUSCEPTIBLE = 0
INFECTED = 1

##########################################################
def simu_intandfire(gin, threshold, tmax, delta):
    """Simplified integrate-and-fire.
    Adapted from @chcomin. If you use this code, please cite:
    'Structure and dynamics: the transition from nonequilibrium to equilibrium in
    integrate-and-fire dynamics', Comin et al., 2012 """
    info(inspect.stack()[0][3] + '()')

    g = gin.copy()
    n = g.vcount()
    initial_charges = np.random.randint(int(threshold*1.3), size=g.vcount())

    edges = np.array(g.get_edgelist())
    par, child = zip(*edges)
    h = np.ones(edges.shape[0])
    A = spa.csr_matrix((h, (child, par)),shape=(n, n))

    acc = initial_charges.copy()

    stats = []
    fires = np.zeros(n, dtype=int)
    for i in range(tmax):
        is_spiking = (acc >= threshold)
        fires += is_spiking
        charge_gain = A.dot(is_spiking)
        acc = acc - acc*is_spiking + charge_gain
        if i % delta == 0:
            stats.append([np.mean(fires), np.std(fires)])
            fires = np.zeros(n, dtype=int)

    return np.array(stats)

##########################################################
def infection_step(adj, status0, beta, gamma):
    """Individual iteration of the SIS transmission/recovery"""

    # Recovery
    status1 = status0.copy()
    randvals = np.random.rand(np.sum(status0))
    lucky = randvals < gamma
    infinds = np.where(status0)[0]
    recinds = infinds[np.where(lucky)]
    status1[recinds] = 0

    # Infection
    status2 = status1.copy()
    q = np.ones(len(adj), dtype=float) - beta # Prob of not being infected, q
    aux = adj[status1.astype(bool), :] # Filter out arcs departing from recovered
    kins = np.sum(aux, axis=0)
    probs = 1 - np.power(q, kins) # Prob of infecting is (1-q^kin)
    posprobids = np.where(probs)[0]
    posprobs = probs[posprobids]
    randvals = np.random.rand(len(posprobs))
    relinds = np.where(randvals < posprobs)
    status2[posprobids[relinds]] = 1
    balance = np.sum(status2) - np.sum(status0)

    return status2, status2 - status1

##########################################################
def set_initial_status(n, i0):
    """Set initial status"""
    status = np.zeros(n, dtype=int)
    choice = np.random.choice(range(n), size=i0, replace=False)
    status[choice] = INFECTED
    return status

##########################################################
def simu_sis(gin, beta, gamma, i0, tmax, delta):
    """Simulate the SIS epidemics model """
    info(inspect.stack()[0][3] + '()')
    adj = np.array(gin.get_adjacency().data)
    n = gin.vcount()

    status = set_initial_status(n, i0)

    stats = []
    ninfections = np.zeros(n, dtype=int)
    for i in range(tmax):
        status, newinf = infection_step(adj, status, beta, gamma)
        ninfections += newinf
        if i % delta == 0:
            stats.append([np.mean(ninfections), np.std(ninfections)])
            ninfections = np.zeros(n, dtype=int)

    return np.array(stats)

##########################################################
def find_closest_factors(n):
    m = int(np.sqrt(n))
    while n % m != 0:
        m -= 1
    return m, n / m

##########################################################
def generate_data(top, n, k):
    """Generate data"""
    info(inspect.stack()[0][3] + '()')
    m = round(k / 2)

    h, w = find_closest_factors(n)

    if top == 'la':
        g = igraph.Graph.Lattice([w, h], nei=1, circular=False)
    elif top == 'er':
        erdosprob = k / n
        g = igraph.Graph.Erdos_Renyi(n, erdosprob)
    elif top == 'ba':
        g = igraph.Graph.Barabasi(n, m)
    elif top == 'ws':
        rewprob = 0.2
        g = igraph.Graph.Lattice([w, h], nei=1, circular=False)
        g.rewire_edges(rewprob)
    elif top == 'gr':
        ngr, r = get_rgg_params(n, k)
        mindiff = 999
        for i in range(3): # Get the graph with closest nvertices
            gnew = igraph.Graph.GRG(ngr, r).clusters().giant()
            if np.abs(gnew.vcount() - n) >= mindiff: continue
            g = gnew
            mindiff = np.abs(g.vcount() - n)
    elif top == 'sb':
        if k == 5: x = 4.5
        elif k == 6: x = 8.3
        elif k == 7: x = 12.5
        elif k == 8: x = 16.2
        pref = (np.array([[14, 1], [1, x]]) / n).tolist()
        n2 = n // 2
        szs = [ n2, n - n2 ]
        g = igraph.Graph.SBM(n, pref, szs, directed=False, loops=False)
    return g

##########################################################
def randomwalk(l, startnode, trans):
    """Random walk assuming a transition matrix with elements such that
    trans[i, j] represents the probability of i going j."""
    n = trans.shape[1]
    walk = - np.ones(l+1, dtype=int)
    walk[0] = startnode
    for i in range(l):
        walk[i+1] = np.random.choice(range(n), p=trans[walk[i], :])
    return walk

##########################################################
def simu_walk(g, walklen, delta):
    """Walk with length @walklen in graph @g and update the @visits and
    @avgdgrees, in the positions given by @idx.
    The first @trimsz of the walk is disregarded."""
    info(inspect.stack()[0][3] + '()')
    adj = np.array(g.get_adjacency().data)
    trans = adj / np.sum(adj, axis=1).reshape(adj.shape[0], -1)
    startnode = np.random.randint(0, g.vcount())
    walkfull = randomwalk(walklen, startnode, trans)

    stats = []
    m = int(walklen / delta)
    for i in range(1, m):
        walk = walkfull[(i-1) * m : i *m]
        vs, cs = np.unique(walk, return_counts=True)
        visits = np.zeros(g.vcount(), dtype=int)
        for v, c in zip(vs, cs):
            visits[v] = c
        stats.append([np.mean(visits), np.std(visits)])

    return np.array(stats)

##########################################################
def generate_conn_graph(top, n, k, maxtries=100):
    """Create a connected graph"""
    info(inspect.stack()[0][3] + '()')
    conn = False
    tries = 0
    while not conn:
        g = generate_data(top, n, k)
        conn = g.is_connected()
        if tries > maxtries:
            raise Exception('Could not find a connected graph')
        tries += 1
    info('{} tries to generate a undirected connected graph'.format(tries))
    return g, tries

##########################################################
def plot_graph(g, top, outdir):
    """Plot graph"""
    if top in ['gr', 'wx']:
        aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
    else:
        if top in ['la', 'ws']:
            layoutmodel = 'grid'
        else:
            layoutmodel = 'fr'
        aux = np.array(g.layout(layoutmodel).coords)
    coords = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0)) # minmax

    f = pjoin(outdir, 'graph_und.png')
    igraph.plot(g, f, layout=coords.tolist())

#############################################################
def get_rgg_params(n, avgdegree):
    rggcatalog = {
        '600,6': [628, 0.0562]
    }

    k = '{},{}'.format(n, avgdegree)
    if k in rggcatalog.keys(): return rggcatalog[k]

    def f(r):
        g = igraph.Graph.GRG(n, r)
        return np.mean(g.degree()) - avgdegree

    return n, scipy.optimize.brentq(f, 0.0001, 10000)

def plot_stats(stats, lbl, nepochs, delta, outdir):
    # W = 640; H = 480
    # fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    # ax.scatter(x, y)
    # outpath = pjoin(outdir, lbel + '.png')
    # plt.savefig(outpath)

    figscale = 8
    fig, axs = plt.subplots(1, 2, squeeze=True,
                figsize=(2*figscale, figscale))

    n = len(stats)
    xs = np.arange(n) * delta
    ylbls = ['Mean', 'Std']
    for jj in range(2):
        axs[jj].plot(xs, stats[:, jj])
        axs[jj].set_xlabel('Time')
        axs[jj].set_ylabel('{} over {} steps'.format(ylbls[jj], delta))
    plt.tight_layout()
    outpath = pjoin(outdir, lbl + '.png')
    plt.savefig(outpath); plt.close()

##########################################################
def main(outdir):
    top = 'er'
    n = 600
    k = 6
    seed = 0
    ei0 = .2
    ebeta = .7
    egamma = .5
    fthresh = 6
    np.random.seed(seed); random.seed(seed)

    gorig, tries = generate_conn_graph(top, n, k)
    gorig.to_directed()

    g = gorig.copy()
    i0 = int(ei0*n)

    nepochs = 1000000
    delta = 100

    t0 = time.time()
    walkstats  = simu_walk(g, nepochs, delta)
    plot_stats(walkstats, 'nvisits', nepochs, delta, outdir)
    np.save(pjoin(outdir, 'walkstats.npy'), walkstats)

    t1 = time.time()
    firestats = simu_intandfire(g, fthresh, nepochs, delta)
    plot_stats(firestats, 'nfires', nepochs, delta, outdir)
    np.save(pjoin(outdir, 'firestats.npy'), firestats)

    t2 = time.time()
    infecstats = simu_sis(g, ebeta, egamma, i0, nepochs, delta)
    plot_stats(infecstats, 'ninfections', nepochs, delta, outdir)
    np.save(pjoin(outdir, 'infecstats.npy'), firestats)

    t3 = time.time()

    return t1-t0, t2-t1, t3-t2

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    dt1, dt2, dt3 = main(args.outdir)
    timetxt = 'Times:\nWalk:\t{:.02f}\nFire:\t{:.02f}\nSIS:\t{:.02f}\n'.format(dt1, dt2, dt3)
    open(readmepath, 'a').write(timetxt)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

