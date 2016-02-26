import glob
import sys
from math import sqrt

if len(sys.argv)<2:
    sys.exit("python performance_significance.py directory")

dirtocheck = sys.argv[1]
test_matrix=[[] for k in range(10)]

seeds_cv=[[] for k in range(10)]
seeds_test=[[] for k in range(10)]
seeds_std_dev=[[] for k in range(10)]

for s in range(42,52):
    cv_values=[0]*10
    test_values=[0]*10
    std_dev=[0]*10
    filename=[[] for i in xrange(10)]

    for radius in range(9):
        for l in [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]:
            for C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
# MKL                
#        for f_name in glob.iglob(dirtocheck+"/*.seed"+str(s)+".L"+str(l)):
#                kODDSTCv1.r4.l1.1.seed44.c1000.0
                for f_name in glob.iglob(dirtocheck+"/*.r"+str(radius)+".l"+str(l)+".seed"+str(s)+".c"+str(C)):
                    f = open(f_name, 'r')
                    i=-3

                    for line in f:
                        i+=1
                        if (i>=0):
                            words=line.split()[0:3]
                            words=map(float,words)

                            if cv_values[i]<words[0]:
                                cv_values[i] = words[0]
                                test_values[i]=words[1]
                                std_dev[i]=words[2]
                                filename[i]=f_name

    cv_string=str(zip(cv_values,test_values,std_dev,filename))
    cv_string=cv_string.replace("[(", "")
    cv_string=cv_string.replace(")]", "")
    cv_string=cv_string.replace("), ", "\n")
    cv_string=cv_string.replace("(", "")
    cv_string=cv_string.replace(",", "")
    cv_avg = sum(cv_values) / float(len(cv_values))
    test_avg = sum(test_values) / float(len(test_values))
    std_dev_avg = sum(std_dev) / float(len(std_dev))
    seeds_cv[s-42]=cv_avg
    seeds_test[s-42]=test_avg
    seeds_std_dev[s-42]=std_dev_avg
    test_matrix[s-42]=test_values
    print("SEED "+str(s)+"\n")
    print cv_string
    print ("AVERAGE "+str(cv_avg)+"\t"+str(test_avg)+"\t"+str(std_dev_avg))

seeds_cv_avg = sum(seeds_cv) / float(len(seeds_cv))
seeds_test_avg = sum(seeds_test) / float(len(seeds_test))
average = sum(seeds_test) / len(seeds_test)
variance = sum((average - value) ** 2 for value in seeds_test) / len(seeds_test)
std_deviation = sqrt(variance)
seeds_std_dev_avg = sum(seeds_std_dev) / float(len(seeds_std_dev))
print ("\t \t INN_CV_AVERAGE \t TEST_AVERAGE \t STD_DEV \t AVG_STD_DEV_SINGLE_FOLDS(??)")

print ("SEEDS AVERAGE "+str(seeds_cv_avg)+"\t"+str(seeds_test_avg)+"\t"+str(std_deviation)+"\t"+str(seeds_std_dev_avg))

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
