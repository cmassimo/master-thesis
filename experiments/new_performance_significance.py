import glob
import sys
from math import sqrt
import numpy as np

if len(sys.argv)<2:
    sys.exit("python performance_significance.py directory")

dirtocheck = sys.argv[1]
outfile = sys.argv[2]
k = 10
#test_matrix=[[] for k in range(10)]

seeds_cv = np.zeros(k)
seeds_test = np.zeros(k)
seeds_std_dev = np.zeros(k)

output = open(outfile, 'w')

for s in range(42,52):
    cv_values = np.zeros(k)
    test_values = np.zeros(k)
    std_dev = np.zeros(k)
    filename=['' for i in range(k)]

    # multikernel.seed46.L0.8
    for f_name in glob.iglob(dirtocheck+"/*.seed"+str(s)+".L0.[0-7]"):
        try:
            f = np.loadtxt(f_name, delimiter="\t", skiprows=2, dtype='str')
        except StopIteration:
            output.write("error in:" + f_name+"\n")
            raise

        for i, line in enumerate(f):
            fline = np.array(line[0:3], dtype='float64')
            if cv_values[i] < fline[0]:
                cv_values[i] = fline[0]
                test_values[i] = fline[1]
                std_dev[i] = fline[2]
                filename[i] = f_name

    output.write("SEED "+str(s)+"\n")

    for cs in zip(cv_values,test_values,std_dev,filename):
        output.write(" ".join(map(lambda x: str(x), cs))+"\n")

    cv_avg = cv_values.mean()
    test_avg = test_values.mean()
    std_dev_avg = std_dev.mean()
    seeds_cv[s-42]=cv_avg
    seeds_test[s-42]=test_avg
    seeds_std_dev[s-42]=std_dev_avg
    #test_matrix[s-42]=test_values
    output.write("AVERAGE "+str(cv_avg)+"\t"+str(test_avg)+"\t"+str(std_dev_avg)+"\n")

seeds_cv_avg = seeds_cv.mean()
seeds_test_avg = seeds_test.mean()
average = seeds_test.mean()
std_deviation = seeds_test.std()
seeds_std_dev_avg = seeds_std_dev.mean()

output.write("\n\t\tINN_CV_AVERAGE\tTEST_AVERAGE\tTEST_STD_DEV\tAVG_STD_DEV_SINGLE_FOLDS(??)\n")
output.write("SEEDS AVERAGE\t"+str(seeds_cv_avg)+"\t"+str(seeds_test_avg)+"\t"+str(std_deviation)+"\t"+str(seeds_std_dev_avg)+"\n")
output.close()

"""
print("MATRIX FOR SIGNIFICATIVITY TEST (1 ROW PER SEED)")
s = [[str(e) for e in row] for row in test_matrix]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print '\n'.join(table)
#print test_matrix
#print test_string
"""                     
