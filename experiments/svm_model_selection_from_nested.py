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
maxparam=(0,0,0,0)
maxacc=0
maxdev=0
cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for h in range(1,11):
    for l in [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]:
        for s in range(42,52):
            for c in cs:
                accuracies=[]
                # kODDSTCv1.r3.l1.3.seed45.c0.0001
                for f_name in glob.iglob(dirtocheck+"/*.r"+str(h)+".l"+str(l)+".seed"+str(s)+".c"+str(c)):
                #for f_name in glob.iglob(dirtocheck+"/*.r"+str(h)+".seed"+str(s)+".c"+str(c)):
                    try:
                        f = np.loadtxt(f_name, delimiter="\t", skiprows=2, dtype='float_')
                    except StopIteration:
                        print "error in:", f_name
                        raise

                    for i, line in enumerate(f):
                        fline = line[0:3]
                        accuracies.append(fline[1])

                accuracies = np.array(accuracies)
                if maxacc < accuracies.mean():
                    maxacc=accuracies.mean()
                    maxparam=(h,l,s,c)
                    maxdev=accuracies.std()
                    print maxacc
                    print maxparam

print "params selected:", maxparam
print "accuracy:", maxacc, "std:", maxdev
