import glob
import sys
from math import sqrt
import numpy as np

if len(sys.argv)<2:
    sys.exit("python performance_significance.py directory")

datasets = ['AIDS', 'CAS', 'CPDB', 'GDD', 'NCI1']
fname = sys.argv[1]
if fname == 'wlc':
    col = 2
else:
    col = 3
times = [[] for i in range(3)]

for dset in datasets:
    res = np.loadtxt("/home/cmassimo/tesi/times/"+dset+"/svm_odd"+fname+"v1.csv", delimiter=',', skiprows=1)
    times[0].append(res[:,col].sum())
    res = np.loadtxt("/home/cmassimo/tesi/times/"+dset+"/mkl_"+fname+".csv", delimiter=',', skiprows=1)
    times[1].append(res[:8,2].sum())
    res = np.loadtxt("/home/cmassimo/tesi/times/"+dset+"/mkl_"+fname+"bc.csv", delimiter=',', skiprows=1)
    times[2].append(res[:8,2].sum())
    
for t in times:
    print ",".join(map(lambda x: str(x), t))
