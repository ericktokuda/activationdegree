# Dynamics and directionality
We analyze the influence of the directionality of the graph over the dynamics that occurrs on top of it. We are currently considering three types of dynamics: random walk, simplified integrate-and-fire, and the SIS epidemic model.

There are two main files:
 * dynamics.py
 * analyze.py

The execution of dynamics.py requires a configuration file, in json format.

## Output
One simulation of each dynamics is performed on each graph. The graph changes as we remove edges. Thus, we end up evaluating the dynamics on nbatches+1 graphs (+1 comes from the unaltered graph). 


It generates:
 * degrees.npy : vertex degrees for each graph
 * lfires.npy : number of fires in the last step for each graph
 * linfec.npy : number of infections in the last step for each graph
 * nattempts.npy : number of attempts required to (1) generate a connected graph and (2) find edges to remove while keeping the graph strongly connected. In the lucky case (every attempt requires just one trial), the minimum number here is batchsz + 1.
 * vfires.npy : number of fires per vertex and per graph
 * vinfec.npy : number of infec per vertex and per graph
 * vvisit.npy : number of visits per vertex and per graph
 * infperepoch.npy* : number of infected every X steps, for each graph (may not be generated)

Consider running this code to analyze these files:

```
featnames = ['degrees', 'lfires', 'linfec', 'nattempts', 'vfires', 'vinfec', 'vvisit']
feats = {}
for f in featnames:
    feats[f] = np.load(open(f + '.npy', 'rb'))
```

## Analysis
The results can be analyzed using the analyze.py script.
