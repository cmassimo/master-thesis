import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import colorConverter
from matplotlib.ticker import LinearLocator, StrMethodFormatter
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.colors import LightSource
from matplotlib import cm
rc('font', **{'family':'serif','serif':['Palatino'], 'size': 8})
rc('text', usetex=True)
sys.argv.append("../../cluster_backup/dists/GDD_wl")
sys.argv.append("WL")

plt.clf()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# MKL matrices weights distribution
weights_raw = np.loadtxt(sys.argv[1], delimiter=',', dtype='str').T
kernel = sys.argv[2]
method = sys.argv[3]
dataset = sys.argv[4]
wdict = {re.search('[A-Z]*.h\d+.*', w[0]).group(0):np.array(w[1:], dtype='float64') for w in weights_raw}
Lambdas = [l/10. for l in range(8)]
nmat = len(wdict)
h=10

xs = np.array([np.array(range(nmat), dtype='float_') for i in range(len(Lambdas))])
ys = np.array([np.zeros(nmat)+(i/10.) for i in range(len(Lambdas))])

plots = []
labels = []
tops = []
ticks = ['' for i in range(nmat)]
topjs = []
wlt=[]
wls=[]
for i, L in enumerate(Lambdas):
    counter = 0
    wlt = []
    wls = []
    weights = []
    for r in range(h+1):
        ck = kernel+".h10.depth"+str(r)
        cw = wdict[ck][i]
        weights.append(cw)
        wls.append("h:"+str(h)+"\nd:"+str(r))
        wlt.append(counter)
        counter +=1
    labels.append("$\Lambda$ "+str(L))
    plots.append(np.array(weights, dtype='float64'))
#    print sum(weights)

x = xs
y = ys
z = plots
#print x.shape, y.shape, len(z)
#ax.set_yticks(range(nmat),map(lambda x: str(x), np.array(range(12))/10.))
ax.view_init(20,-50)
ax.view_init(10,-115)
ax.set_xticks(wlt)
ax.set_xticklabels(wls)
ax.set_yticks(Lambdas)
ax.set_yticklabels(Lambdas)

cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/len(Lambdas), alpha=0.5) for i in range(len(Lambdas))]
for i in range(nmat):
    colors.append((0,0,0,.35))
surf = ax.plot_wireframe(x, y, z, rstride=1, cstride=1, colors=colors, linewidth=1, antialiased=True)

ax.set_xlabel("kernel parameters")
ax.set_xlim3d(0, nmat)
ax.set_ylabel("$\Lambda$")
ax.set_ylim3d(0, 0.7)
ax.set_zlabel('kernel weight')
ax.set_zlim3d(0, 0.13)
#ax.set_zlim3d(0, 0.15)
plt.title("Kernel weights distribution\n for $("+kernel+")^{"+method+"}$ on the "+dataset+" dataset")
plt.tight_layout()
#plt.show()
plt.savefig("kca_"+dataset+"_"+kernel+"_"+method+".pdf")
