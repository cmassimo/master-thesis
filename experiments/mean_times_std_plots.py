import sys
import numpy as np
#from skgraph.datasets import load_graph_datasets
#from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
#from skgraph.kernel.ODDSTOrthogonalizedGraphKernel import ODDSTOrthogonalizedGraphKernel
import matplotlib.pyplot as plt
from matplotlib import rc
from math import log
from sklearn.preprocessing import scale, normalize
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# media e std tempi tra parametri (INC)
ks = [1,2,4,8,16,32,64,110]
x = np.array(range(1,len(ks)+1))
y = [[] for i in range(len(ks))]
stds = [[] for i in range(len(ks))]

plt.clf()
width=0.2
p0 = plt.bar(x, y[0], width, color='c')
p1 = plt.bar(x+width, y[1], width, color='m')
sdlower = np.maximum(1e-2, stds[1] - (stds[1]-y[1]))
plt.errorbar(x+width/2,y[0], yerr=stds[0], capthick=2, linestyle="None", color='red')
plt.errorbar(x+width*(3/2.), y[1], yerr=[sdlower, stds[1]], capthick=2, linestyle="None", color='green')
#plt.errorbar(x+width*(3/2.), y[1], yerr=stds[1], linestyle="None")
#p0 = plt.plot(x, y[0], color='c')
#p1 = plt.plot(x, y[1], color='m')
#plt.errorbar(x, y[0], yerr=stds[0], linestyle="None")
#plt.errorbar(x, y[1], yerr=stds[1], linestyle="None")
plt.xticks(np.array(range(1,len(ks)+1))+width, ks)#, size='small')
#plt.xticks(np.array(range(1,len(ks))), ks)#, size='small')
#plt.yscale('log')
plt.ylabel('mean time in seconds (and std)')
plt.xlabel('number of kernels (matrices)')
plt.title('Computation times breakdown analysis')
plt.legend((p0[0], p1[0]), ('$(TCK_{ST})^c$', '$(TCK_{ST})^{hs}$'), loc='upper left')
#plt.show()
plt.savefig('mean_times_incremental.pdf')


# AIDS STC
y[0] = np.array([1002.57,1003.97,963.46,1076.79,1029.54,1215.71,1332.18,1456.39])
stds[0] = np.array([138.46,145.87,134.64,96.19,154.82,210.91,222.25,209.04])
y[1] = np.array([86.3875,174.55125,364.56375,756.30125,1259.155,1732.74875,2586.735,3731.96875])
stds[1] = np.array([177.249530024,363.104928376,756.710324694,1572.65645562,2500.80105002,3017.83099442,3761.82709733,4567.92014267])

# AIDS STPC
y[0] = np.array([])
stds[0] = np.array([])
y[1] = np.array([])
stds[1] = np.array([])

# CPDB STPC
y[0] = np.array([])
stds[0] = np.array([])
y[1] = np.array([])
stds[1] = np.array([])
