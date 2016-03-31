import sys
import numpy as np
#from skgraph.datasets import load_graph_datasets
#from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
#from skgraph.kernel.ODDSTOrthogonalizedGraphKernel import ODDSTOrthogonalizedGraphKernel
import matplotlib.pyplot as plt
from matplotlib import rc
from math import log
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

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

#x = np.array([15.97398, 28.09628, 12.086722, 2.134117, 299.721692])
x = np.array([12.086722, 15.97398, 2.134117, 299.721692, 28.09628])
#x1 = np.array([32.597597, 58.937005, 26.602846, 4.154025, 660.222288])
x1 = np.array([26.602846, 32.597597, 4.154025, 660.222288, 58.937005])
ratios = x/x1
#x1 = map(lambda y: log(y), x1)
#x = [x1[i]*ratios[i] for i in range(len(x1))]
bins = np.array(range(5))
#ks = ['CAS', 'NCI1', 'AIDS', 'CPDB', 'GDD']
ks = ['AIDS', 'CAS', 'CPDB', 'GDD', 'NCI1']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
width = 0.2
p0 = plt.bar(bins-0.1, x, width, color='m', align='center', tick_label=ks)
p1 = plt.bar(bins, x1, width, color='c', align='edge', tick_label=ks)
ax.set_ylabel('time in seconds')
ax.set_xlabel('dataset')
ax.set_yscale('log')
ax.set_title('Kernel computation performances')
ax.legend((p0[0], p1[0]), ('incremental', 'sequential'), loc='upper left')
plt.show()

plt.savefig('kernel_times.pdf')

# media e std tempi tra parametri
#x = np.array([range(1+i,13+i+1,3) for i in range(3)])
x = np.array(range(1,6))
y = [[] for i in range(3)]
stds = [[] for i in range(3)]
#ks = ['', '', '1503 (AIDS)', '', '', '4337 (CAS)', '', '', '684 (CPDB)', '', '','1178 (GDD)', '', '', '4110 (NCI1)', '', '']
#ks = ['', '', '4337 (CAS)', '', '', '4110 (NCI1)', '', '', '1503 (AIDS)', '', '','1178 (GDD)', '', '', '684 (CPDB)', '', '']
ks = ['4337 (CAS)', '4110 (NCI1)', '1503 (AIDS)', '1178 (GDD)', '684 (CPDB)']
ks = [k for k in reversed(ks)]
indices2=[2, 3, 0, 4, 1] 
y[0] = np.array([1206.69, 25545.21, 160.81, 474.45, 21271.12]) 
stds[0] = np.array([148.27, 3550.58, 17.02, 56.17, 3095.15])
y[1] = np.array([1384.66, 27240.70, 191.20, 536.91, 22898.62])
stds[1] = np.array([172.73, 3345.26, 18.01, 59.30, 3778.88])
y[2] = np.array([3731.97, 36011.18, 976.86, 412.32, 33020.56])
stds[2] = np.array([4567.92, 42964.59, 1172.59, 115.27, 37706.40])
plt.clf()
width=0.2
p0 = plt.bar(x, y[0][indices2], width, color='c')
p1 = plt.bar(x+width, y[1][indices2], width, color='m')
p2 = plt.bar(x+2*width, y[2][indices2], width, color='y')
plt.errorbar(x+width/2,y[0][indices2],yerr=stds[0][indices2], linestyle="None")
plt.errorbar(x+width*(3/2.),y[1][indices2],yerr=stds[1][indices2], linestyle="None")
plt.errorbar(x+width*(5/2.),y[2][indices2],yerr=stds[2][indices2], linestyle="None")
plt.xticks(np.array(range(1,6))+(3/2.)*width, ks)#, size='small')
plt.ylabel('mean time in seconds (and std)')
plt.xlabel('number of records (dataset)')
plt.title('Methods total computation times analysis')
plt.legend((p0[0], p1[0], p2[0]), ('MKL ($TCK_{ST}$ {\small orthogonalized})', 'MKL ($TCK_{ST}$)', 'SVM ($TCK_{ST}$)'), loc='upper left')
plt.show()

# plot tempi totali.
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
width=0.25
x1 = np.array(range(1,4))
x2 = np.array(range(1,3))
y = [[] for i in range(3)]
ks = ['1503\n(AIDS)', '4337\n(CAS)', '684\n(CPDB)', '1178\n(GDD)', '4110\n(NCI1)']
indices=[2, 3, 0]
indices2=[4, 1] 
#y[0] = np.array(map(lambda x: x, [13273.57, 280997.32, 1768.88, 5219, 233982.28]))
#y[1] = np.array(map(lambda x: x, [15231.23, 299647.71, 2103.2, 5906.0099, 251884.87]))
#y[2] = np.array(map(lambda x: x, [29855.75, 288089.46, 7814.8899, 3298.54, 264164.47]))
# WL TIMES
y[0] = np.array([8402.91,0.0,911.18,3036.13,160413.38])
y[1] = np.array([8730.86,0.0,1030.77,3008.87,172659.05])
y[2] = np.array([2757.74,0.0,710.85,1195.20,0.0])
plt.clf()
plt0 = plt.subplot(1, 2, 1)
#plt.suptitle('\Large Nested 10-fold cross-validation times\n \large for the three methods using $TCK_{ST}$')
#plt.suptitle('\Large Nested 10-fold cross-validation times\n \large for the three methods using $TCK_{ST+}$')
plt.suptitle('\Large Nested 10-fold cross-validation times\n \large for the three methods using the $WLC$ kernel')
p0 = plt0.bar(x1+2*width, y[0][indices], width, color='c')
p1 = plt0.bar(x1+width, y[1][indices], width, color='m')
p2 = plt0.bar(x1, y[2][indices], width, color='y')
plt.xticks(np.array(range(1,4))+(3/2.)*width, np.array(ks)[indices])#, size='small')
plt.ylabel('total time in seconds')
plt.xlabel('number of samples (dataset)')
width=0.15
plt1 = plt.subplot(1, 2, 2)
p0 = plt1.bar(x2+2*width, y[0][indices2], width, color='c')
p1 = plt1.bar(x2+width, y[1][indices2], width, color='m')
p2 = plt1.bar(x2, y[2][indices2], width, color='y')
plt.xticks(np.array(range(1,3))+(3/2.)*width, np.array(ks)[indices2])#, size='small')
plt.xlabel('number of samples (dataset)')
#plt0.legend((p2[0], p1[0], p0[0]), ('$(TCK_{ST})^{hs}$', '$(TCK_{ST})^c$', '$(TCK_{ST})^{oc}$'), loc='upper left', borderaxespad=.5)
#plt0.legend((p2[0], p1[0], p0[0]), ('$(TCK_{ST+})^{hs}$', '$(TCK_{ST+})^c$', '$(TCK_{ST+})^{oc}$'), loc='upper left', borderaxespad=.5)
plt0.legend((p2[0], p1[0], p0[0]), ('$(WLC)^{hs}$', '$(WLC)^c$', '$(WLC)^{oc}$'), loc='upper left', borderaxespad=.5)
plt0.grid(True, axis='y')
plt1.grid(True, axis='y')
plt.show()

#ST TIMES
y[0] = np.array([1347.84,10235.5,4027.45,182243.92,217526.46])
y[1] = np.array([1601.27,411799.46,552.36,198607.11,229147.23])
y[2] = np.array([7814.8899,329855.75,298.54,264164.4699,288089.45999])

#STP TIMES
y[0] = np.array([10383.54,0.0,1435.74,4463.23,0.0])
y[1] = np.array([11528.67,233809.02,1884.27,4641.18,194816.36])
y[2] = np.array([21109.49,226882.42,5477.92,3503.45,0.0])

# WL TIMES
y[0] = np.array([8402.91,0.0,911.18,3036.13,160413.38])
y[1] = np.array([8730.86,0.0,1030.77,3008.87,172659.05])
y[2] = np.array([2757.74,0.0,710.85,1195.20,0.0])
