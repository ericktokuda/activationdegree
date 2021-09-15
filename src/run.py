#!/usr/bin/env python3
"""Run experiments for the directed graphs survey
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
import scipy.sparse as spa


##########################################################
def int_and_fire(gin, threshold, tmax, trimsz):
    """Simplified integrate-and-fire.
    Adapted from @chcomin. If you use this code, please cite:
    'Structure and dynamics: the transition from nonequilibrium to equilibrium in
    integrate-and-fire dynamics', Comin et al., 2012
    """

    g = gin.copy()
    n = g.vcount()
    initial_charges = np.random.randint(int(threshold*1.3), size=g.vcount())

    edges = np.array(g.get_edgelist())
    par, child = zip(*edges)
    h = np.ones(edges.shape[0])
    A = spa.csr_matrix((h, (child, par)),shape=(n, n))

    acc = initial_charges.copy()

    for i in range(trimsz):
        is_spiking = (acc >= threshold)
        charge_gain = A.dot(is_spiking)
        acc = acc - acc*is_spiking + charge_gain

    fires = np.zeros(n, dtype=int)
    for i in range(tmax-trimsz):
        is_spiking = (acc >= threshold)
        fires += is_spiking
        charge_gain = A.dot(is_spiking)
        acc = acc - acc*is_spiking + charge_gain

    return fires

##########################################################
def generate_data(top, n, k):
    """Generate data"""
    info(inspect.stack()[0][3] + '()')
    m = round(k / 2)
    width = int(np.sqrt(n))
    if top == 'la':
        g = igraph.Graph.Lattice([width, width], nei=1, circular=False)
    elif top == 'er':
        erdosprob = k / n
        g = igraph.Graph.Erdos_Renyi(n, erdosprob)
    elif top == 'ba':
        g = igraph.Graph.Barabasi(n, m)
    elif top == 'ws':
        rewprob = 0.2
        g = igraph.Graph.Lattice([width, width], nei=1, circular=False)
        g.rewire_edges(rewprob)
    elif top == 'gr':
        radius = 3 # radius = get_rgg_params(n, k)
        g = igraph.Graph.GRG(n, radius)
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
def remove_arc_conn(g):
    """Remove and arc while keep the graph strongly connected"""
    edgeids = np.arange(0, g.ecount())
    np.random.shuffle(edgeids)

    for eid in edgeids:
        newg = g.copy()
        newg.delete_edges([g.es[eid]])
        if newg.is_connected(mode='strong'):
            return newg
    raise Exception()

##########################################################
def evaluate_walk(idx, g, walklen, trimsz):
    """Walk with length @walklen in graph @g and update the @visits and @avgdgrees,
    in the positions given by @idx. The first @trimsz of the walk is disregarded."""
    adj = np.array(g.get_adjacency().data)
    trans = adj / np.sum(adj, axis=1).reshape(adj.shape[0], -1)
    startnode = np.random.randint(0, g.vcount())
    walk = randomwalk(walklen, startnode, trans)
    vs, cs = np.unique(walk[trimsz:], return_counts=True)
    visits = np.zeros(g.vcount(), dtype=int)
    for v, c in zip(vs, cs):
        visits[v] = c
    return visits

##########################################################
def run_experiment(gorig, nsteps, batchsz, walklen, ifirethresh, ifireepochs, trimrel):
    """Remove @batchsz arcs, @nsteps times and evaluate a walk of len
    @walklen and the integrate-and-fire dynamics"""
    # info(inspect.stack()[0][3] + '()')
    g = gorig.copy()
    wtrim = int(walklen * trimrel)
    ftrim = int(ifireepochs * trimrel)

    shp = (nsteps+1, g.vcount())
    err = - np.ones(shp, dtype=int)
    wvisits = np.zeros(shp, dtype=float)
    # wvisits = np.random.randint(0, 100, size=shp)
    nfires = np.zeros(shp, dtype=int)
    avgdegrees = np.zeros(shp, dtype=int)

    avgdegrees[0, :] = g.degree(mode='out')
    wvisits[0, :] = evaluate_walk(0, g, walklen, wtrim)
    nfires[0, :] = int_and_fire(g, ifirethresh, ifireepochs, ftrim)

    for i in range(nsteps):
        info('Step {}'.format(i))
        for _ in range(batchsz):
            try: g = remove_arc_conn(g)
            except: raise Exception('Could not remove arc in step {}'.format(i))
        avgdegrees[i+1, :] = g.degree(mode='out')
        # wvisits[i+1, :] = evaluate_walk(i+1, g, walklen, trimsz) / g.vcount()
        wvisits[i+1, :] = evaluate_walk(i+1, g, walklen, wtrim)
        nfires[i+1, :] = int_and_fire(g, ifirethresh, ifireepochs, ftrim)
    return avgdegrees, wvisits, nfires

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

    igraph.plot(g, pjoin(outdir, 'graph_und.png'), layout=coords.tolist())

##########################################################
def initial_check(nepochs, batchsz, g):
    """Check whether too many arcs are being removed."""
    if (nepochs * batchsz) > (0.75 * g.ecount()):
        info('Consider altering nepochs, batchsz, and avgdegree.')
        info('Execution may fail.')
        raise Exception('Too may arcs to be removed.')
    elif not g.is_connected(mode='strong'):
        raise Exception('Initial graph is not strongly connected.')

##########################################################
def plot_visits_degree(visits, degrees, outpath):
    """Plot the number of visits by the degree for each vertex.
    Each dot represent an vertex"""
    # info(inspect.stack()[0][3] + '()')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    p = pearsonr(degrees, visits)[0]
    ax.scatter(degrees, visits)
    ax.set_title('Pearson {:.03f}'.format(p))
    ax.set_xlabel('Vertex degree')
    ax.set_ylabel('Number of visits')
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_correlation_degree(meas, label, degrees, outpath):
    """Plot the number of visits by the degree for each vertex.
    Each dot represent an vertex"""
    # info(inspect.stack()[0][3] + '()')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    p = pearsonr(degrees, meas)[0]
    ax.scatter(degrees, meas)
    ax.set_title('Pearson {:.03f}'.format(p))
    ax.set_xlabel('Vertex degree')
    ax.set_ylabel(label)
    plt.savefig(outpath)
    plt.close()

##########################################################
def plot_correlations(wvisits, nfires, degrees, outdir, filesuff):
    woutpath = pjoin(outdir, 'w_' + filesuff + '.png')
    foutpath = pjoin(outdir, 'f_' + filesuff + '.png')
    # plot_correlation_degree(wvisits, 'Walk visits', degrees, woutpath)
    plot_correlation_degree(nfires, 'Number of fires', degrees, foutpath)

##########################################################
def main(cfg):
    np.random.seed(cfg.seed); random.seed(cfg.seed)
    stronglyconn = False
    maxtries = 100
    tries = 0

    while not stronglyconn:
        g = generate_data(cfg.top, cfg.nvertices, cfg.avgdegree)
        stronglyconn = g.is_connected(mode='strong')
        if tries > maxtries:
            raise Exception('Could not find strongly connected graph')
        tries += 1
    info('{} tries to generate a strongly connected graph'.format(tries))

    plot_graph(g, cfg.top, cfg.outdir)
    g.to_directed()

    initial_check(cfg.nepochs, cfg.batchsz, g)

    retshp = (cfg.nrealizations, cfg.nepochs + 1, g.vcount())
    wvisits = - np.ones(retshp, dtype=int)
    nfires = - np.ones(retshp, dtype=int)
    degrees = - np.ones(retshp, dtype=int)

    for r in range(cfg.nrealizations):
        info('Realization {}'.format(r))
        kd, wv, nf = run_experiment(g, cfg.nepochs, cfg.batchsz, cfg.walklen,
                                cfg.ifirethresh, cfg.ifireepochs, cfg.trimrel)
        wvisits[r, :, :] = wv
        nfires[r, :, :] = nf
        degrees[r, :] = kd

    np.save(pjoin(cfg.outdir, 'degrees.npy'), degrees)
    np.save(pjoin(cfg.outdir, 'wvisits.npy'), wvisits)
    np.save(pjoin(cfg.outdir, 'nfires.npy'), nfires)

    for i in range(wvisits.shape[0]): # realization
        for k in range(wvisits.shape[1]): # epoch
            filesuff = '{:02d}_{:04d}'.format(i, k)
            plot_correlations(wvisits[i, k, :], nfires[i, k, :], degrees[i, k, :],
                              cfg.outdir, filesuff)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='config in json format')
    args = parser.parse_args()

    cfg = json.load(open(args.config),
            object_hook=lambda d: SimpleNamespace(**d))
    os.makedirs(cfg.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, cfg.outdir)
    shutil.copy(args.config, pjoin(cfg.outdir, 'config.json'))
    main(cfg)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(cfg.outdir))
