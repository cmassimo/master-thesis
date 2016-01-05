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
for r in [1, 2, 3, 4]:
    for it in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]:
        for l in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0"]:
            for C in ['0.01','0.1','1.0','10.0','100.0']:
                print (r,it,l,C)
                accuracies=[]
                for f_name in glob.iglob(dirtocheck+"/*.r"+str(r)+".i"+it+".l"+l+"*.c"+C):
#                    print f_name
                    f = open(f_name,'r')
                    i=-3
                    for line in f:
                        i+=1
                        if (i>=0):
                            words=line.split()
                            words=map(float,words)
                            accuracies.append(words[1])
                    f.close()
    #                if i!=9:
    #                    print "missing line ",f_name
                #print "len acc", len(accuracies) 
                accuracies = np.array(accuracies)
		newacc = accuracies.mean()
                if maxacc < newacc:
                    maxacc=newacc
                    maxparam=(r,it,l,C)
                    maxdev=accuracies.std()

print "max param, ",maxparam
print "accuracy, ",maxacc, " std: ",maxdev

