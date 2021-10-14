# Dynamics and directionality
We analyze the influence of the directionality of the graph over the dynamics that occurrs on top of it. We are currently considering three types of dynamics: random walk, simplified integrate-and-fire, and the SIS epidemic model.

There are two main files:
 * dynamics.py
 * analyze.py

The execution of dynamics.py requires a configuration file, in json format.

## Output
One simulation of each dynamics is performed on each graph. The graph changes as we remove edges.

It generates:
 * degrees.npy : vertex degrees
 * infperepoch.npy : number of infected every X steps, for each graph
 * lfires.npy : number of fires in the last step for each graph
 * linfec.npy : number of infections in the last step for each graph
 * nattempts.npy : number of attempts required to (1) generate a connected graph and (2) find edges to remove while keeping the graph strongly connected. The minimum number here is batchsz + 1
 * vfires.npy : number of fires per vertex and per graph
 * vinfec.npy : number of infec per vertex and per graph

## Analysis
The results can be analyzed using the analyze.py script.
