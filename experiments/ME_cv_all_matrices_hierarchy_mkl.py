from joblib import Parallel, delayed
import re
import sys, os
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from cvxopt import matrix
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.FastEasyMKL import FastEasyMKL
from ME_innerCV_easyMKL_hierarchy import calculate_inner_AUC_kfold, single_kernel_train_and_sum

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile shape kernels matrix_files*")

L = float(sys.argv[1])
outfile = sys.argv[2]
shape = int(sys.argv[3])
kernels =  sys.argv[4].split(',')
folds = 10 

matrix_files = {}
for i in range(len(kernels)):
    matrix_files[i] = [mf for mf in sys.argv[5:len(sys.argv)] if re.match(".*"+kernels[i]+"\..*", mf)]

km, target_array = load_svmlight_file(sys.argv[5], shape, zero_based=True)

for rs in range(42,52):
    f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')
    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

    f.write("Total examples "+str(km.shape[0])+"\n")
    f.write("CV\t\t test_roc_score\n")
    del km

    # OUTER K-FCV
    for train_index, test_index in kf:
        y_train = target_array[train_index]

        # COMPUTE INNER K-FOLD
        inner_scores = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape, tr_index=train_index)

        f.write(str(inner_scores.mean())+"\t")

        start = time.clock()
        grams = Parallel(n_jobs=2)(delayed(single_kernel_train_and_sum)(L, target_array, folds, rs, matrix_files[i], shape) for i in matrix_files.keys())
        end = time.clock()
        print "END sum matrices calculation, elapsed time: %0.4f s" % (end - start)

        start = time.clock()

        train_grams=[]
        test_grams=[]
        tr_i = matrix(train_index)
        te_i = matrix(test_index)

        for i in range(len(grams)):
            train_grams.append(grams[i][tr_i,tr_i])
            test_grams.append(grams[i][te_i,tr_i])

        easy = EasyMKL(lam=L, tracenorm = True)
        easy.train(train_grams, matrix(y_train))

        end = time.clock()

        del train_grams
        
        # predict on test examples
        y_test = target_array[test_index]
        ranktest = np.array(easy.rank(test_grams))
        rte = roc_auc_score(np.array(y_test), ranktest)
        print "END Training, elapsed time: %0.4f s" % (end - start)

        f.write(str(rte)+"\t")
        f.write(str(inner_scores.std())+"\t")
        f.write(",".join(map(lambda x: str(x), easy.weights)) + "\n")
        del test_grams
        del easy

    f.close()

