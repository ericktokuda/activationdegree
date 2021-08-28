#!/usr/bin/env python3
"""Run experiments for the directed graphs survey
"""

import argparse
import time, datetime
import os, random
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import igraph
from myutils import info, create_readme

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
        igraph.plot(g, outpath)
    elif top == 'ba':
        g = igraph.Graph.Barabasi(n, m)
        igraph.plot(g, outpath)
    elif top == 'ws':
        rewprob = 0.2
        g = igraph.Graph.Lattice([width, width], nei=1, circular=False)
        g.rewire_edges(rewprob)
        igraph.plot(g, outpath)
    elif top == 'gr':
        radius = 3 # radius = get_rgg_params(n, k)
        g = igraph.Graph.GRG(n, radius)
        igraph.plot(g, outpath)
    g.to_directed()
    return g

##########################################################
def randomwalk(l, startnode, trans):
    """Random walk assuming a transition matrix with elements such that
    trans[i, j] represents the probability of i going j."""
    # info(inspect.stack()[0][3] + '()')
    n = trans.shape[1]
    walk = [startnode]
    for i in range(l):
        pos = walk[-1]
        newpos = np.random.choice(range(n), p=trans[pos, :])
        walk.append(newpos)
    return walk

##########################################################
def remove_arc_conn(g):
    """Remove and arc while keep the graph strongly connected"""
    # info(inspect.stack()[0][3] + '()')
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
    g = gorig.copy()
    countsall = np.zeros((nsteps, g.vcount()), dtype=int)
    for i in range(nsteps):
        for _ in range(batchsz):
            newg, succ = remove_arc_conn(g)
            if not succ: return [], False
            else: g = newg
        adj = np.array(g.get_adjacency().data)
        trans = adj / np.sum(adj, axis=1).reshape(adj.shape[0], -1)

        startnode = np.random.randint(0, g.vcount())
        #TODO: perform multiple walks for each intermediate graph?
        walk = randomwalk(walklen, startnode, trans)
        vs, cs = np.unique(walk, return_counts=True)
        
        for v, c in zip(vs, cs):
            countsall[i, v] = c
    return countsall, True

##########################################################
def plot_graph(g, top, outdir):
    """Plot graph"""
    if top == 'la':
        igraph.plot(g, pjoin(outdir, top + '.png'), layout='grid')
    else:
        igraph.plot(g, pjoin(outdir, top + '.png'), layout='fr')

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
def main(seed, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    np.random.seed(seed); random.seed(seed)

    n = 500
    k = 5 # nvertcies ~= (k * n) / 2
    nepochs = 10
    batchsz = 3
    nrealizations = 3
    top = 'la'
    walklen = 100

    outpath = pjoin(outdir, 'la.pdf')

    g = generate_data(top, n, k)
    plot_graph(g, top, outdir)
    initial_check(nepochs, batchsz, g)

    countsall = - np.ones((nrealizations, nepochs, g.vcount()), dtype=int)
    for r in range(nrealizations):
        ret, succ = run_experiment(g, nepochs, batchsz, walklen)
        if not succ: info('DECIDE WHAT TO DO')
        else: countsall[r, :, :] = ret
    
    breakpoint()
    
    return

    # g = g.clusters().giant()

    if topologymodel in ['gr', 'wx']:
        aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
    else:
        if topologymodel in ['la', 'ws']:
            layoutmodel = 'grid'
        else:
            layoutmodel = 'random'
        aux = np.array(g.layout(layoutmodel).coords)

    coords = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0)) # minmax

    info('For Aiur!')

    # comeca com rede nao dirigida
    # sorteia  uma ligacao e apga uma aresta dirigida (uma das duas)
    # manter a componente fortemente conexacorrelacao com grau de ativacao
    # ER WS BA spatial
    # 2000 nodes
    # <k> = 5


##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.seed, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
