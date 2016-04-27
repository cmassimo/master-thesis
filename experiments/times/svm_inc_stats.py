import glob
import sys
from math import sqrt
import numpy as np
from sklearn.preprocessing import normalize

if len(sys.argv)<2:
    sys.exit("python performance_significance.py directory")

fname = sys.argv[1]

cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
nmat = [1, 2, 4, 8, 16, 32, 64, 110]
k = 8

results = np.loadtxt(fname, delimiter=',', skiprows=1)

times = dict.fromkeys(nmat)
for n in nmat:
    times[n] = dict.fromkeys(cs)

# kODDSTCv1.r10.l1.1.seed46.c0.1
for c in cs:
    for n in nmat:
        times[n][c] = np.array([r[1] for r in results[:,2:] if r[0] == c])[:n].sum()

means = []
stds = []
for n in nmat:
    means.append(normalize([np.array(times[n].values())])[0].mean())
    stds.append(normalize([np.array(times[n].values())])[0].std())

for m in means:
    print m

print "---"

for s in stds:
    print s

