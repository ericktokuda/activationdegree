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
import scipy
import scipy.sparse as spa
from multiprocessing import Pool
import pandas as pd

##########################################################
SUSCEPTIBLE = 0
INFECTED = 1

##########################################################
def simu_intandfire(gin, threshold, tmax, trimsz):
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

    return fires, np.sum(is_spiking)

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
    posprobids = np.where(probs)[0] # Disregard vtx if it has no infectious neigh
    posprobs = probs[posprobids]
    randvals = np.random.rand(len(posprobs))
    relinds = np.where(randvals < posprobs)
    status2[posprobids[relinds]] = 1
    # balance = np.sum(status2) - np.sum(status0)
    return status2, status2 - status1

##########################################################
def set_initial_status(n, i0):
    """Set initial status"""
    status = np.zeros(n, dtype=int)
    choice = np.random.choice(range(n), size=i0, replace=False)
    status[choice] = INFECTED
    return status

##########################################################
def simu_sis(gin, beta, gamma, i0, trimsz, tmax):
    """Simulate the SIS epidemics model """
    adj = np.array(gin.get_adjacency().data)
    n = gin.vcount()

    status = set_initial_status(n, i0)

    for i in range(trimsz):
        status, _ = infection_step(adj, status, beta, gamma)

    ninfections = np.zeros(n, dtype=int)
    for i in range(tmax-trimsz):
        if (i % 10000) == 0: info('SIS step {}'.format(i))
        status, newinf = infection_step(adj, status, beta, gamma)
        ninfections += newinf
        j = i + trimsz

    return ninfections, np.sum(newinf)

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
def remove_two_arcs_conn(g, premoval, pedges, maxtries=100):
    """Remove two arcs, while keeping @g strongly connected. If @premoval
    is true, the arcs are removed in pair, in which case a random edge is
    taken from @pedges. Return the new graph and the attempts to remove"""
    
    ids = []
    if premoval:
        pedges2  = pedges.copy()
        np.random.shuffle(pedges2)
        
        for v1, v2 in pedges2:
            e1 = g.get_eid(v1, v2) # forward arc
            e2 = g.get_eid(v2, v1) # backward arc
            ids.append((e1, e2))
    else:
        alledges = list(range(0, g.ecount())) # Two random arcs from all arcs
        for i in range(maxtries):
            ids.append(random.sample(alledges, k=2)) # No reposition

    ntries = 1
    for e1, e2 in ids:
        newg = g.copy()
        newg.delete_edges([e1, e2])
        if newg.is_connected(mode='strong'):
            arcs = [(g.es[e1].source, g.es[e1].target),
                    (g.es[e2].source, g.es[e2].target)]
            return newg, arcs, ntries
        ntries += 1
    raise Exception()

##########################################################
def simu_walk(idx, g, walklen, trimsz):
    """Walk with length @walklen in graph @g and update the @visits and
    @avgdgrees, in the positions given by @idx.
    The first @trimsz of the walk is disregarded."""
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
def update_pedges(arcs, pedges):
    """Update the list of paired edges, @pedges, by removing the @arcs."""
    pair1 = (np.min(arcs[0]), np.max(arcs[0]))
    pair2 = (np.min(arcs[1]), np.max(arcs[1]))

    if pair1 in pedges: pedges.remove(pair1)
    if pair2 in pedges: pedges.remove(pair2)
    return pedges

##########################################################
def generate_conn_graph(top, n, k, maxtries=100):
    """Create a connected graph"""
    info(inspect.stack()[0][3] + '()')
    conn = False
    tries = 0
    while not conn:
        if os.path.exists(top):
            g = igraph.Graph.Read(top)
        else:
            g = generate_data(top, n, k)
        conn = g.is_connected()
        if tries > maxtries:
            raise Exception('Could not find a connected graph')
        tries += 1
    info('{} tries to generate a undirected connected graph'.format(tries))
    return g, tries

##########################################################
def get_pairs_conn_vertices(g):
    """Get every pair of vertices with an arc in-between. The returned 2-uple
    is ordered by vertexid. For example: [(2, 5), (6, 10)]"""
    pedges = set()
    for e in g.es:
        vs = [e.source, e.target]
        pedges.add((np.min(vs), np.max(vs)))
    return list(pedges)

##########################################################
def run_experiment_lst(params):
    return run_experiment(*params)

##########################################################
def run_experiment(top, nreq, kreq, degmode, nbatches, minrecipr,
                   paired, trimrel, fthresh,
                   ei0, ebeta, egamma, outrootdir, seed):
    """Removes @batchsz arcs, @nbatches times and evaluate three different
    dynamics.  Calculates the correlation between vertex in-degree and the
    level of activity """
    np.random.seed(seed); random.seed(seed)

    outdir = pjoin(outrootdir, '{:02d}'.format(seed))
    os.makedirs(outdir, exist_ok=True)

    gorig, tries = generate_conn_graph(top, nreq, kreq) # g is connected
    plot_graph(gorig, top, outdir)
    gorig.to_directed() # g is strongly connected
    n = gorig.vcount()
    narcs = gorig.ecount()
    k = narcs / gorig.vcount() * 2
    # In the ideal case, with no repeated:
    ntoremove =  np.ceil((1 - minrecipr) * narcs).astype(int)
    rem = ntoremove % nbatches
    if rem != 0: ntoremove = (int(ntoremove / nbatches) + 1) * nbatches
    if (ntoremove % 2) != 0: ntoremove += 1 # Paired removal
    batchsz =  int(ntoremove / nbatches)

    nattempts = np.zeros(nbatches + 1, dtype=int)
    nattempts[0] = tries

    initial_check(nbatches, batchsz, gorig)
    g = gorig.copy()
    nepochs = n * 1000
    trim = int(nepochs * trimrel)
    # wtrim = int(nepochs * trimrel)
    # ftrim = int(nepochs * trimrel)
    # etrim = int(nepochs * trimrel)
    wtrim = ftrim = etrim = trim
    i0 = int(ei0*n)

    shp = (nbatches+1, g.vcount())
    err = - np.ones(shp, dtype=int)
    vvisit = np.zeros(shp, dtype=int) # Vertex visits
    vfires = np.zeros(shp, dtype=int) # Vertex fires
    vinfec = np.zeros(shp, dtype=int) # Vertex infections
    degrees = np.zeros(shp, dtype=int)
    lfires = - np.ones(nbatches + 1, dtype=int) # Last step fires
    linfec = - np.ones(nbatches + 1, dtype=int) # Last step inf

    degrees[0, :] = g.degree(mode=degmode)
    vvisit[0, :] = simu_walk(0, g, nepochs, wtrim)
    vfires[0, :], lfires[0] = simu_intandfire(g, fthresh, nepochs, ftrim)
    vinfec[0, :], linfec[0] = simu_sis(g, ebeta, egamma, i0, etrim, nepochs)

    pedges = get_pairs_conn_vertices(g)

    for i in range(nbatches):
        info('Step {}'.format(i))
        for _ in range(batchsz):
            try: g, arcs, m = remove_two_arcs_conn(g, paired, pedges)
            except: raise Exception('Could not remove arc in step {}'.format(i))
            pedges = update_pedges(arcs, pedges)
            nattempts[i+1] += m

        degrees[i+1, :] = g.degree(mode=degmode)
        vvisit[i+1, :] = simu_walk(i+1, g, nepochs, wtrim)
        vfires[i+1, :], lfires[i+1]  = simu_intandfire(g, fthresh, nepochs, ftrim)
        vinfec[i+1, :], linfec[i+1] = simu_sis(g, ebeta, egamma, i0, etrim, nepochs)

    for f in ['degrees', 'vvisit', 'vfires', 'vinfec', 'lfires', 'linfec',
              'nattempts', 'paired']:
        np.save(pjoin(outdir, f + '.npy'), locals()[f])

    corrs = []
    for i in range(nbatches + 1): # nbatches
        c1, c2, c3 = calculate_correlations(vvisit[i, :], vfires[i, :],
                vinfec[i, :], degrees[i, :], i, outdir)
        corrs.append([top, g.vcount(), seed, i, c1, c2, c3, int(paired)])

    return corrs

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

##########################################################
def initial_check(nbatches, batchsz, g):
    """Check whether too many arcs are being removed."""
    if (nbatches * batchsz) > (0.75 * g.ecount()):
        info('Consider altering nbatches, batchsz, and avgdegree.')
        info('Execution may fail.')
        raise Exception('Too may arcs to be removed.')
    elif not g.is_connected(mode='strong'):
        raise Exception('Initial graph is not strongly connected.')

##########################################################
def plot_correlation_degree(meas, label, degrees, p, outpath):
    """Plot the number of visits by the degree for each vertex.
    Each dot represent an vertex"""
    # info(inspect.stack()[0][3] + '()')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(degrees, meas, alpha=.5)
    ax.set_title('Pearson {:.03f}'.format(p))
    ax.set_xlabel('Vertex degree')
    ax.set_ylabel(label)
    plt.savefig(outpath)
    plt.close()

##########################################################
def calculate_correlations(vvisits, vfires, vinfec, degrees, epoch, outdir):
    woutpath = pjoin(outdir, 'w_{:03d}.png'.format(epoch))
    foutpath = pjoin(outdir, 'f_{:03d}.png'.format(epoch))
    eoutpath = pjoin(outdir, 'e_{:03d}.png'.format(epoch))

    c1, c2, c3 = 0, 0, 0

    if np.sum(vvisits):
        wvisitsr = vvisits / np.sum(vvisits)
        c1 = pearsonr(degrees, wvisitsr)[0]
        t1 = 'Relative number of visits'
        plot_correlation_degree(wvisitsr, t1, degrees, c1, woutpath)

    if np.sum(vfires):
        nfiresr = vfires / np.sum(vfires)
        c2 = pearsonr(degrees, nfiresr)[0]
        t2 = 'Relative number of fires'
        plot_correlation_degree(nfiresr, t2, degrees, c2, foutpath)

    if np.sum(vinfec):
        ninfecr = vinfec / np.sum(vinfec)
        c3 = pearsonr(degrees, ninfecr)[0]
        t3 = 'Relative number of infections'
        plot_correlation_degree(ninfecr, t3, degrees, c3, eoutpath)

    return c1, c2, c3

#############################################################
def get_rgg_params(n, avgdegree):
    rggcatalog = {
        '600,6': [628, 0.0562],
        '800,6': [835, 0.0485],
        '1000,6': [1041, 0.0433],
        '3000,6': [3085, 0.0250],
        '5000,6': [5128, 0.01934],
    }

    k = '{},{}'.format(n, avgdegree)
    if k in rggcatalog.keys(): return rggcatalog[k]

    def f(r):
        g = igraph.Graph.GRG(n, r)
        return np.mean(g.degree()) - avgdegree

    return n, scipy.optimize.brentq(f, 0.0001, 10000)

##########################################################
def main(cfg, nprocs):

    np.random.seed(cfg.seed); random.seed(cfg.seed)

    retshp = (cfg.nrealizations, cfg.nbatches + 1, cfg.nvertices)
    seeds = [cfg.seed + i for i in range(cfg.nrealizations)]
    params = []
    for i in range(cfg.nrealizations):
        params.append( [cfg.top, cfg.nvertices, cfg.avgdegree, cfg.degmode,
                        cfg.nbatches, cfg.minrecipr, cfg.paired,
                        cfg.trimrel, cfg.fthresh,
                        cfg.ei0, cfg.ebeta, cfg.egamma,
                        cfg.outdir, seeds[i]] )

    if nprocs == 1:
        corrs = [ run_experiment_lst(p) for p in params ]
    else:
        info('Running in parallel ({})'.format(nprocs))
        pool = Pool(nprocs)
        corrs = pool.map(run_experiment_lst, params)

    data = []

    for i in range(cfg.nrealizations):
        for j in range(cfg.nbatches + 1):
            data.append(corrs[i][j])

    cols = ['top', 'n', 'realiz', 'epoch', 'corrvisits', 'corrfires', 'corrinfec', 'paired']
    pd.DataFrame(data, columns=cols).to_csv(pjoin(cfg.outdir, 'corrs.csv'), index=False)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='config in json format')
    parser.add_argument('--nprocs', default=1, type=int,
            help='Number of processes')
    args = parser.parse_args()

    cfg = json.load(open(args.config), object_hook=lambda d: SimpleNamespace(**d))
    os.makedirs(cfg.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, cfg.outdir)
    shutil.copy(args.config, pjoin(cfg.outdir, 'config.json'))
    main(cfg, args.nprocs)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(cfg.outdir))
