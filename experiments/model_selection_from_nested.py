__author__ = "Riccardo Tesselli"
__date__ = "20/mag/2015"
__credits__ = ["Riccardo Tesselli"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer = "Riccardo Tesselli"
__email__ = "riccardo.tesselli@gmail.com"
__status__ = "Production"

import glob
import sys
import numpy as np

if len(sys.argv)<2:
    sys.exit("python model_selection_from_nested.py directory")

dirtocheck = sys.argv[1]
maxparam=(0,0,0)
maxacc=0
maxdev=0
for L in [i/10. for i in range(11)]:
    accuracies=[]
    for f_name in glob.iglob(dirtocheck+"/*.L"+str(L)):
        try:
            f = np.loadtxt(f_name, delimiter="\t", skiprows=2, dtype='str')
        except StopIteration:
            print "error in:", f_name
            raise

        for i, line in enumerate(f):
            fline = np.array(line[0:3], dtype='float64')
            accuracies.append(fline[1])

    accuracies = np.array(accuracies)
    if maxacc < accuracies.mean():
        maxacc=accuracies.mean()
        maxparam=L
        maxdev=accuracies.std()
print "max param, ",maxparam
print "accuracy, ",maxacc, " std: ",maxdev
