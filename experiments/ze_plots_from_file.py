import sys
import numpy as np
import matplotlib.pyplot as plt
import re

# MKL matrices weights distribution
weights_raw = np.loadtxt("csv/NCI1_mkl_ODDSTC_weights_dist.csv", delimiter=',', dtype='str').T
wdict = {re.search('r\d+\.l\d\.\d', w[0]).group(0):np.array(w[1:], dtype='float64') for w in weights_raw}

lambdas = [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]
Lambdas = [l/10. for l in range(11)]
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/11) for i in range(11)]

plots = []
labels = []
tops = []
ticks = ['' for i in range(110)]
for i, L in enumerate(Lambdas):
    weights = []
    topjs = []
    for r in range(1, 11):
        top = 0
        topj = 0
        topk = None
        for j, l in enumerate(lambdas):
            ck = "r"+str(r)+".l"+str(l)
            cw = wdict[ck][i]
            if cw > top:
                top = cw
                topj = (r*(j+1))-1
                topk = ck
            weights.append(cw)
        topjs.append((top,topj,topk))
    ctop = max(topjs, key=lambda x: x[0])
    tops.append(ctop[1])
    ticks[ctop[1]] = ctop[2]
    plots.append(plt.plot(range(110), np.array(weights, dtype='float64'), color=colors[i]))
    plt.xticks(range(110), ticks, size='small', rotation=90)
    labels.append('L'+str(L))
[plt.axvline(tj, linewidth=0.5, color=colors[i], ls='--') for i, tj in enumerate(tops)]
plt.ylabel('kernel weight')
plt.xlabel('kernel parameters')
plt.title('MKL kernels weights distribution')
plt.legend([p[0] for p in  plots], labels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.show()
#plt.savefig('NCI1_ODDSTC_weights_dist.pdf')

plt.clf()
