import sys
import numpy as np
from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.ODDSTOrthogonalizedGraphKernel import ODDSTOrthogonalizedGraphKernel
import matplotlib.pyplot as plt

ds = load_graph_datasets.load_graphs_CPDB()

radius = 3
lbd = 1
norm = False

ko = ODDSTOrthogonalizedGraphKernel(r=radius, l=lbd, normalization=norm)
keo = ODDSTincGraphKernel(r=radius, l=lbd, normalization=norm, version=1, ntype=0, nsplit=0, kernels=['ODDST'])

buckets = ko.transform_serial_no_matrix(ds.graphs)

feature_lists = keo.transform_serial_explicit_no_matrix(ds.graphs)['ODDST']

x = [len(f) for f in feature_lists.values()]
x1 = [len(f) for f in buckets]
bins = np.array(range(radius+1))

plt.clf()
width = 0.35
p0 = plt.bar(bins-0.175, x, width, color='r', align='center', tick_label=bins)
p1 = plt.bar(bins, x1, width, color='b', align='edge', tick_label=bins)
plt.ylabel('# of features')
plt.xlabel('bucket index')
plt.title('Feature cardinality per bucket')
plt.legend((p0[0], p1[0]), ('enhanced o15n', 'naive o15n'), loc='upper left')
plt.savefig('feats_cardinality.pdf')

allfeats0 = [item[1] for flist in feature_lists.values() for item in flist.items()]
allfeats1 = [item[1] for bucket in buckets for item in bucket.items()]
allfeats0.sort(reverse = True)
allfeats1.sort(reverse = True)

plt.clf()
p0 = plt.plot(range(150), allfeats0[0:150], 'r-')
p1 = plt.plot(range(150), allfeats1[0:150], 'b-')
plt.ylabel('feature weight')
plt.xlabel('feature index')
plt.title('Feature weights distribution')
plt.legend((p0[0], p1[0]), ('enhanced o15n', 'naive o15n'))
#plt.show()
plt.savefig('weights_dist.pdf')

x = [15.97398, 28.09628, 12.086722, 2.134117, 299.721692]
x1 = [32.597597, 58.937005, 26.602846, 4.154025, 660.222288]
bins = np.array(range(5))
ks = ['CAS', 'NCI1', 'AIDS', 'CPDB', 'GDD']

plt.clf()
width = 0.2
p0 = plt.bar(bins-0.1, x, width, color='r', align='center', tick_label=ks)
p1 = plt.bar(bins, x1, width, color='b', align='edge', tick_label=ks)
plt.ylabel('time in seconds')
plt.xlabel('dataset')
plt.title('Kernel computation performances')
plt.legend((p0[0], p1[0]), ('incremental', 'sequential'), loc='upper left')
plt.savefig('kernel_times.pdf')

