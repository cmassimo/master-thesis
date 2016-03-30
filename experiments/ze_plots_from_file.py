import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import colorConverter
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.colors import LightSource
from matplotlib import cm
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
sys.argv.append("../../cluster_backup/partial_weights_dists/sorted/NCI1ststcWD_sorted.csv")
sys.argv.append("ODDSTC,ODDST")

plt.clf()
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111, projection='3d')
#cm = plt.get_cmap('gist_rainbow')
#cm = plt.get_cmap('gist_earth')
#colors = [cm(1.*i/11, alpha=0.5) for i in range(11)]
# MKL matrices weights distribution
weights_raw = np.loadtxt(sys.argv[1], delimiter=',', dtype='str').T
wdict = {re.search('[A-Z]*.r\d+\.l\d\.\d', w[0]).group(0):np.array(w[1:], dtype='float64') for w in weights_raw}
kernels = sys.argv[2].split(',')
lambdas = [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]
Lambdas = [l/10. for l in range(11)]
xs = np.array([np.zeros(220)+i*10 for i in range(11)])
ys = np.array([np.array(range(220), dtype='float_') for i in range(11)])
plots = []
labels = []
tops = []
nmat = len(wdict)
ticks = ['' for i in range(nmat)]
topjs = []
for i, L in enumerate(Lambdas):
    weights = []
    top = 0
    topj = 0
    topk = None
    for k, key in enumerate(kernels):
        for r in range(1, 11):
            for j, l in enumerate(lambdas):
                ck = key+".r"+str(r)+".l"+str(l)
                cw = wdict[ck][i]
                if cw > top:
                    top = cw
                    topj = r*(j+1)+(k*110)-1
                    topk = ck
                weights.append(cw)
    topjs.append((top,topj,topk))
    ctop = max(topjs, key=lambda x: x[0])
    tops.append(ctop[1])
    ticks[ctop[1]] = ctop[2]
    labels.append("$\Lambda$ "+str(L))
#    plots.append(plt.plot(range(nmat), np.array(weights, dtype='float64'), color=colors[i]))
    plots.append(np.array(weights, dtype='float64'))
pass
#[plt.axvline(tj+(i/10), linewidth=0.5, color=colors[i], ls='--') for i, tj in enumerate(tops)]
#fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#ls = LightSource(270, 11)
y = xs
x = ys
z = plots
#rgb = ls.shade(z, cmap=cm, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=10, cstride=1, cmap=cm.coolwarm)#, linewidth=0, antialiased=True, shade=False)
ax.set_xlabel('matrices')
ax.set_xlim3d(0, 220)
ax.set_ylabel("$\Lambda$")
ax.set_ylim3d(0, 110)
ax.set_zlabel('weights')
ax.set_zlim3d(0, 0.03)
plt.show()


poly = PolyCollection(plots, facecolors=colors)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')
ax.set_xlabel('matrices')
ax.set_xlim3d(0, 220)
ax.set_zlabel('weights')
ax.set_zlim3d(0, 0.03)
ax.set_ylabel("$\Lambda$")
ax.set_ylim3d(0, 11)


plt.ylabel('kernel weight')
plt.xlabel('kernel parameters')
plt.title('MKL kernels weights distribution')
plt.legend([p[0] for p in  plots], labels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)


#plt.savefig('NCI1_ODDSTC_weights_dist.pdf')
