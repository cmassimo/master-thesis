import sys, os
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from cvxopt import matrix
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.FastEasyMKL import FastEasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile shape matrix_files*")

L = float(sys.argv[1])
outfile = sys.argv[2]
shape = int(sys.argv[3])

matrix_files = sys.argv[4:len(sys.argv)]

grams = []
start = time.clock()

for mf in matrix_files:
    km, target_array = load_svmlight_file(mf, shape, zero_based=True)
    grams.append(km.todense())

end = time.clock()

print "Matrices loaded in: " + str(end - start)

# CONF VARS
folds = 10
random_states = range(42,52)

for rs in random_states:
    f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')

    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

    f.write("Total examples "+str(km.shape[0])+"\n")
    f.write("CV\t\t test_roc_score\n")

    # OUTER K-FCV
    for train_index, test_index in kf:

        y_train = target_array[train_index]

        # COMPUTE INNER K-FOLD
        #print "Computing inner "+str(folds)+"FCV..."
        inner_scores = calculate_inner_AUC_kfold([grams[i][np.ix_(train_index, train_index)] for i in range(len(grams))], y_train, l=L, rs=rs, folds=folds)
        #print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

        f.write(str(inner_scores.mean())+"\t")

        #print "Outer training..."
        start = time.clock()

        easy = FastEasyMKL(lam=L, tracenorm = True)
        easy.train([matrix(grams[i][np.ix_(train_index, train_index)]) for i in range(len(grams))], matrix(y_train))

        end = time.clock()
        print "Outer training, elapsed time:", (end - start)

        y_test = target_array[test_index]
        
        # predict on test examples
        ranktest = np.array(easy.rank([matrix(grams[i][np.ix_(test_index, train_index)]) for i in range(len(grams))]))
        del easy

        rte = roc_auc_score(np.array(y_test), ranktest)

        f.write(str(rte)+"\t")
        
        f.write(str(inner_scores.std())+"\n")

    f.close()

#print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

