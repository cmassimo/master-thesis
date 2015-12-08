import sys
import numpy as np
from skgraph.kernel.ODDSTPCGraphKernel import ODDSTPCGraphKernel
from skgraph.datasets import load_graph_datasets
import matplotlib.pyplot as plt

g_it = load_graph_datasets.load_graphs_bursi().graphs
kernel = ODDSTPCGraphKernel(r=3, l=1, normalization=False)


res = {}

for g in g_it:
    tmp = kernel.getFeaturesNoCollisionsExplicit(g).items()
    for (k,v) in tmp:
        if res.get(k) == None:
            res[k] = v
        else:
            res[k] += v


data = np.array(res.values())
data.sort()
data = data[::-1]
plt.plot(data[20:1000])
plt.show()
