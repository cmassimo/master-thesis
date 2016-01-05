# CIDM paper replication

import sys, os
import time
import numpy as np
from cvxopt import matrix
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

if len(sys.argv)<4:
    sys.exit("python cross_validation_from_matrices_wl.py matrix_file1 matrix_file2 L outfile")

matrix_file1 = sys.argv[1]
matrix_file2 = sys.argv[2]
L = float(sys.argv[3])
outfile = sys.argv[4]

# esempio: ODDSTmkl.dCAS.r3.l1.3.L0.5.svmlight
km1, target_array = load_svmlight_file(matrix_file1+".svmlight")
km2, target_array = load_svmlight_file(matrix_file2+".svmlight")

km1 = km1[:,1:].todense().tolist()
km2 = km2[:,1:].todense().tolist()
grams = [km1, km2]

# CONF VARS
folds = 3
random_states = [42]#range(42,52)

sc=[]

for rs in random_states:
    f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')

    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)
    
    f.write("Total examples "+str(len(km1[0]))+"\n")
    f.write("CV\t\t test_roc_score\n")
    
    # OUTER K-FCV
    for train_index, test_index in kf:
        train_grams=[]
        test_grams=[]
        
        for i in range(len(grams)):
            train_grams.append([])
            test_grams.append([])

            index=-1    
            for row in grams[i]:
                index+=1    
                if index in train_index:
                    train_grams[i].append(np.array(row).take(train_index))
                else:
                    test_grams[i].append(np.array(row).take(train_index))

        y_train = target_array[train_index]
        y_test = target_array[test_index]

        # COMPUTE INNER K-FOLD
        print "Computing inner "+str(folds)+"FCV..."
        inner_scores = calculate_inner_AUC_kfold(train_grams, y_train, l=L, rs=rs, folds=folds)
        print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

        f.write(str(inner_scores.mean())+"\t")

        for i in xrange(len(train_grams)):
            train_grams[i]=matrix(np.array(train_grams[i]))

        for i in xrange(len(test_grams)):
            test_grams[i]=matrix(np.array(test_grams[i]))

        print "Training..."
        start = time.clock()

        easy = EasyMKL(lam=L, tracenorm = True)
        easy.train(train_grams, matrix(y_train))

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)
        
        # predict on test examples
        ranktest = np.array(easy.rank(test_grams))
        rte = roc_auc_score(np.array(y_test), ranktest)
    
        f.write(str(rte)+"\t")
        
        f.write(str(inner_scores.std())+"\n")

    f.close()

scores=np.array(sc)

print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

