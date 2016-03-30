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
x1 = map(lambda y: math.log(y), x1)
x = [x1[i]*ratios[i] for i in range(len(x1))]
bins = np.array(range(5))
#ks = ['CAS', 'NCI1', 'AIDS', 'CPDB', 'GDD']
ks = ['AIDS', 'CAS', 'CPDB', 'GDD', 'NCI1']

plt.clf()
width = 0.2
p0 = plt.bar(bins-0.1, x, width, color='m', align='center', tick_label=ks)
p1 = plt.bar(bins, x1, width, color='c', align='edge', tick_label=ks)
plt.ylabel('time in seconds (log)')
plt.xlabel('dataset')
plt.title('Kernel computation performances')
plt.legend((p0[0], p1[0]), ('incremental', 'sequential'), loc='upper left')
#plt.show()

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

# media e std tempi tra parametri
#x = [range(1+i,13+i+1,3) for i in range(3)]
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#width=0.2
#x = np.array(range(1,6))
#y = [[] for i in range(3)]
#stds = [[] for i in range(3)]
##ks = ['', '', '1503 (AIDS)', '', '', '4337 (CAS)', '', '', '684 (CPDB)', '', '','1178 (GDD)', '', '', '4110 (NCI1)', '', '']
##ks = ['', '', '4337 (CAS)', '', '', '4110 (NCI1)', '', '', '1503 (AIDS)', '', '','1178 (GDD)', '', '', '684 (CPDB)', '', '']
#ks = ['4337\n(CAS)', '4110\n(NCI1)', '1503\n(AIDS)', '1178\n(GDD)', '684\n(CPDB)']
#ks = [k for k in reversed(ks)]
##indices=[6, 3, 4, 1, 5, 2, 1, 0]
#indices2=[2, 3, 0, 4, 1] 
#y[0] = np.array(map(lambda x: log(x), [1206.69, 25545.21, 160.81, 474.45, 21271.12])) 
#stds[0] = np.array(map(lambda x: log(x), [148.27, 3550.58, 17.02, 56.17, 3095.15]))
#y[1] = np.array(map(lambda x: log(x), [1384.66, 27240.70, 191.20, 536.91, 22898.62]))
#stds[1] = np.array(map(lambda x: log(x), [172.73, 3345.26, 18.01, 59.30, 3778.88]))
#y[2] = np.array(map(lambda x: log(x), [3731.97, 36011.18, 976.86, 412.32, 33020.56]))
#stds[2] = np.array(map(lambda x: log(x), [4567.92, 42964.59, 1172.59, 115.27, 37706.40]))
#plt.clf()
#p0 = plt.bar(x, y[0][indices2], width, color='c')
#p1 = plt.bar(x+width, y[1][indices2], width, color='m')
#p2 = plt.bar(x+2*width, y[2][indices2], width, color='y')
#plt.errorbar(x+width/2,y[0][indices2],yerr=stds[0][indices2], linestyle="None")
#plt.errorbar(x+width*(3/2.),y[1][indices2],yerr=stds[1][indices2], linestyle="None")
#plt.errorbar(x+width*(5/2.),y[2][indices2],yerr=stds[2][indices2], linestyle="None")
#plt.xticks(np.array(range(1,6))+(3/2.)*width, ks)#, size='small')
##plt.xticks(range(17), ks)#, size='small')
#plt.ylabel('mean time and std in seconds (log)')
#plt.xlabel('number of records (dataset)')
#plt.title('Methods computation times analysis\n\large for the three methods using $TCK_{ST}$')
#plt.legend((p0[0], p1[0], p2[0]), ('MKL ($TCK_{ST}$ {\small orthogonalized})', 'MKL ($TCK_{ST}$)', 'SVM ($TCK_{ST}$)'), loc='upper left')
#plt.show()

# plot tempi totali.
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#width=0.2
#x = np.array(range(1,6))
#y = [[] for i in range(3)]
#ks = ['1503\n(AIDS)', '4337\n(CAS)', '684\n(CPDB)', '1178\n(GDD)', '4110\n(NCI1)']
#indices=[2, 3, 0, 4, 1]
#indices2=[2, 3, 0, 4, 1] 
#y[0] = np.array(map(lambda x: log(x), [13273.57, 280997.32, 1768.88, 5219, 233982.28]))
#y[1] = np.array(map(lambda x: log(x), [15231.23, 299647.71, 2103.2, 5906.0099, 251884.87]))
#y[2] = np.array(map(lambda x: log(x), [29855.75, 288089.46, 7814.8899, 3298.54, 264164.47]))
#plt.clf()
#p0 = plt.bar(x, y[0][indices2], width, color='c')
#p1 = plt.bar(x+width, y[1][indices2], width, color='m')
#p2 = plt.bar(x+2*width, y[2][indices2], width, color='y')
#plt.xticks(np.array(range(1,6))+(3/2.)*width, np.array(ks)[indices])#, size='small')
#plt.ylabel('total time in seconds (log)')
#plt.xlabel('number of records (dataset)')
#plt.title('\Large Nested 10-fold cross-validation times\n \large for the three methods using $TCK_{ST}$')
#plt.legend((p0[0], p1[0], p2[0]), ('MKL ($TCK_{ST}$ {\small orthogonalized})', 'MKL ($TCK_{ST}$)', 'SVM ($TCK_{ST}$)'), loc='upper left')
#plt.show()

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
y[0] = np.array(map(lambda x: x, [10235.5,217526.46, 1347.84, 4027.45,182243.92]))
y[1] = np.array(map(lambda x: x, [11799.46,229147.23,1601.27,4552.36,198607.11]))
y[2] = np.array(map(lambda x: x, [29855.75,288089.45999,7814.8899,3298.54,264164.4699]))
plt.clf()
plt0 = plt.subplot(1, 2, 1)
plt.suptitle('\Large Nested 10-fold cross-validation times\n \large for the three methods using $TCK_{ST}$')
p0 = plt0.bar(x1, y[0][indices], width, color='c')
p1 = plt0.bar(x1+width, y[1][indices], width, color='m')
p2 = plt0.bar(x1+2*width, y[2][indices], width, color='y')
plt.xticks(np.array(range(1,4))+(3/2.)*width, np.array(ks)[indices])#, size='small')
plt.ylabel('total time in seconds')
plt.xlabel('number of records (dataset)')
width=0.15
plt1 = plt.subplot(1, 2, 2)
p0 = plt1.bar(x2, y[0][indices2], width, color='c')
p1 = plt1.bar(x2+width, y[1][indices2], width, color='m')
p2 = plt1.bar(x2+2*width, y[2][indices2], width, color='y')
plt.xticks(np.array(range(1,3))+(3/2.)*width, np.array(ks)[indices2])#, size='small')
#plt.ylabel('total time in seconds')
plt.xlabel('number of records (dataset)')
#plt.figlegend((p0[0], p1[0], p2[0]),
#        ('MKL ($TCK_{ST}$ {\small orthogonalized})', 'MKL ($TCK_{ST}$)', 'SVM ($TCK_{ST}$)'),
#        loc='upper left', ncol=3, mode="expand", borderaxespad=1.)
plt0.legend((p0[0], p1[0], p2[0]), ('\small MKL ($TCK_{ST}$ {\scriptsize orthogonal})', '\small MKL ($TCK_{ST}$)', '\small SVM ($TCK_{ST}$)'), loc='upper left', borderaxespad=.5)
plt.show()
