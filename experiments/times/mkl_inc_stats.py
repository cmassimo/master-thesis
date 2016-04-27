import glob
import sys
from math import sqrt
import numpy as np

if len(sys.argv)<2:
    sys.exit("python performance_significance.py directory")

mfname = sys.argv[1]
sfname = sys.argv[2]

#ODD
nmat = [1, 2, 4, 8, 16, 32, 64, 110]
#GDD
#nmat = [1, 2, 4, 8, 16, 32]
#WLC
#nmat = [1, 2, 4, 8]
k=2
#k=1
kern='_stp'
#kern='_wlc'
times = []

# kODDSTCv1.r10.l1.1.seed46.c0.1
for i, n in enumerate(nmat):
    results = np.loadtxt(mfname+"/incremental"+kern+str(n), delimiter=',', skiprows=1)
    times.append(results[:,2])

cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
results = np.loadtxt(sfname, delimiter=',', skiprows=1)

# kODDSTCv1.r10.l1.1.seed46.c0.1

for i, n in enumerate(nmat):
    times.append(np.array([np.array([r[1] for r in results[:,k:] if r[0] == c])[:n].sum() for c in cs]))

for t in times:
    print ",".join(map(lambda x: str(x),t))
