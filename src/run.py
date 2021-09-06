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
            return newg, True
    return g, False

##########################################################
def run_experiment(gorig, nsteps, batchsz, walklen):
    """Removal of arcs and evaluation of walks."""
    info(inspect.stack()[0][3] + '()')
    # trimsz = 0
    trimsz = int(walklen * 0.2) # Discard initial trimsz
    g = gorig.copy()
    shp = (nsteps+1, g.vcount())
    err = - np.ones(shp, dtype=int)
    visits = np.zeros(shp, dtype=int)
    avgdegrees = np.zeros(shp, dtype=int)

    # Walk on the original graph
    adj = np.array(g.get_adjacency().data)
    trans = adj / np.sum(adj, axis=1).reshape(adj.shape[0], -1)
    startnode = np.random.randint(0, g.vcount())
    walk = randomwalk(walklen, startnode, trans)
    vs, cs = np.unique(walk[trimsz:], return_counts=True)
    for v, c in zip(vs, cs): visits[0, v] = c
    avgdegrees[0, :] = g.degree(mode='out')

    for i in range(nsteps):
        info('Step {}'.format(i))
        for _ in range(batchsz):
            newg, succ = remove_arc_conn(g)
            if succ:
                g = newg
            else:
                info('Could not remove arc in step {}'.format(i))
                return err, err
        adj = np.array(g.get_adjacency().data)
        trans = adj / np.sum(adj, axis=1).reshape(adj.shape[0], -1)

        startnode = np.random.randint(0, g.vcount())
        walk = randomwalk(walklen, startnode, trans)
        vs, cs = np.unique(walk[trimsz:], return_counts=True)
        for v, c in zip(vs, cs):
            visits[i+1, v] = c
        avgdegrees[i+1, :] = g.degree(mode='out')
    return visits, avgdegrees

##########################################################
def plot_graph(g, top, outdir):
    """Plot graph"""
    if top in ['gr', 'wx']:
        aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
    else:
        if top in ['la', 'ws']:
            layoutmodel = 'grid'
        else:
            layoutmodel = 'random'
        aux = np.array(g.layout(layoutmodel).coords)
    coords = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0)) # minmax

    igraph.plot(g, pjoin(outdir, top + '.png'), layout=coords.tolist())

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
    ax.set_title('Pearson {:.02f}'.format(p))
    ax.set_xlabel('Vertex degree')
    ax.set_ylabel('Number of visits')
    plt.savefig(outpath)
    plt.close()

##########################################################
def main(cfg):
    np.random.seed(cfg.seed); random.seed(cfg.seed)
    stronglyconn = False
    maxtries = 20
    tries = 0

    while not stronglyconn:
        g = generate_data(cfg.top, cfg.nvertices, cfg.avgdegree)
        stronglyconn = g.is_connected(mode='strong')
        if tries > maxtries:
            raise Exception('Could not find strongly connected graph')
        tries += 1

    g.to_directed()

    plot_graph(g, cfg.top, cfg.outdir)
    initial_check(cfg.nepochs, cfg.batchsz, g)

    retshp = (cfg.nrealizations, cfg.nepochs + 1, g.vcount())
    visits = - np.ones(retshp, dtype=int)
    degrees = - np.ones(retshp, dtype=int)

    for r in range(cfg.nrealizations):
        info('Realization {}'.format(r))
        vd, kd = run_experiment(g, cfg.nepochs, cfg.batchsz, cfg.walklen)
        visits[r, :, :] = vd
        degrees[r, :] = kd

    np.save(pjoin(cfg.outdir, 'visits.npy'), visits)
    np.save(pjoin(cfg.outdir, 'degrees.npy'), degrees)

    for i in range(visits.shape[0]): # realization
        for k in range(visits.shape[1]): # epoch
            f = '{:02d}_{:04d}.png'.format(i, k)
            outpath = pjoin(cfg.outdir, f)
            plot_visits_degree(visits[i, k, :], degrees[i, k, :], outpath)

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
