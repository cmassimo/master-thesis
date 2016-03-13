import glob
import sys
from math import sqrt
import numpy as np

if len(sys.argv)<2:
    sys.exit("python performance_significance.py directory keys")

dirtocheck = sys.argv[1]
keys_file = sys.argv[2]
size = int(sys.argv[3])

keys = np.loadtxt(keys_file, dtype='str')[0:110]

wd = {key:np.zeros(11) for key in keys}

for i, l in enumerate(map(lambda x: x/10.0, range(11))):
    for f_name in glob.iglob(dirtocheck+"/*.L"+str(l)):
        f = np.loadtxt(f_name, delimiter="\t", skiprows=2, dtype='str')

        for line in f:
            weights = np.array(line[3].split(','), dtype='float64')
            for j, key in enumerate(keys):
                wd[key][i] += weights[j]

f = open(dirtocheck+"/weight_dist.csv", 'w')
f.write(",".join(keys))
f.write("\n")
for i in range(11):
    f.write(",".join([str(wd[k][i]/100.0) for k in keys]))
    f.write("\n")

f.close()
