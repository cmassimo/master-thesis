import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import colorConverter
from matplotlib.ticker import LinearLocator, IndexLocator, StrMethodFormatter
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.colors import LightSource
from matplotlib import cm
rc('font', **{'family':'serif','serif':['Palatino'], 'size': 8})
rc('text', usetex=True)
#sys.argv.append("../../cluster_backup/dists/CAS_mklst")
#sys.argv.append("ODDST")

plt.clf()

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# MKL matrices weights distribution
weights_raw = np.loadtxt(sys.argv[1], delimiter=',', dtype='str').T
kernel = sys.argv[2]
method = sys.argv[3]
dataset = sys.argv[4]
maxheight = int(sys.argv[5])
wdict = {re.search('[A-Z]*.r\d+.*', w[0]).group(0):np.array(w[1:], dtype='float64') for w in weights_raw}
Lambdas = [l/10. for l in range(8)]
nmat = len(wdict)
h=range(1,maxheight)

xs = np.array([np.array(range(nmat), dtype='float_') for i in range(len(Lambdas))])
ys = np.array([np.zeros(nmat)+(i/10.) for i in range(len(Lambdas))])

plots = []
labels = []
tops = []
ticks = ['' for i in range(nmat)]
topjs = []

wls = []
wlt= []
for i, L in enumerate(Lambdas):
    weights = []
    wls= []
    wlt=[]
    counter = 0
    for r in h:
        for j, l in enumerate(range(r+1)):
            ck = kernel+".r"+str(r)+".depth"+str(l)
            cw = wdict[ck][i]
            weights.append(cw)
            if r==l:
                wls.append("h:"+str(r)+"\nd:"+str(l))
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
#ll=LinearLocator(len(Lambdas))
#ll.tick_values(0.0,1.0)
#ax.w_yaxis.set_major_locator(ll) 
ax.view_init(10,-115)
ax.set_xticks(wlt)
ax.set_xticklabels(wls)
ax.set_yticks(Lambdas)
ax.set_yticklabels(Lambdas)

cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/len(Lambdas), alpha=0.5) for i in range(len(Lambdas))]
for i in range(nmat):
    colors.append((0,0,0,.35))
#cm = plt.get_cmap('gist_earth')
surf = ax.plot_wireframe(x, y, z, rstride=1, cstride=1, colors=colors, linewidth=1, antialiased=True)
#surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=cm.coolwarm)

ax.set_xlabel('kernel parameters')
ax.set_xlim3d(0, nmat)
ax.set_ylabel("$\Lambda$")
ax.set_ylim3d(0, 0.7)
ax.set_zlabel('kernel weight')
ax.set_zlim3d(0, 0.14)
#ax.set_zlim3d(0, 0.30)
plt.title("Kernel weights distribution\n for $("+kernel+")^{"+method+"}$ on the "+dataset+" dataset")
plt.tight_layout()
plt.show()
#plt.savefig("kca_"+dataset+"_"+kernel+"_"+method+".pdf")
