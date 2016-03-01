from joblib import Parallel, delayed
import time
import numpy as np
from cvxopt import matrix, mul
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from skgraph.kernel.EasyMKL.FastEasyMKL import FastEasyMKL
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file

# K-FOLD CROSS VALIDATION
def calculate_inner_AUC_kfold(Y, l, rs, folds, mfiles, shape, tr_index):
    out_index = matrix(tr_index)
    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    for train_index, test_index in kf:
        start = time.clock()
        grams = Parallel(n_jobs=2)(delayed(single_kernel_train_and_sum)(l, train_index, Y, folds, rs, mfiles[i], shape, out_index) for i in mfiles.keys())
#        grams.append(single_kernel_train_and_sum(l, train_index, Y, folds, rs, mfiles[0], shape, out_index))
        end = time.clock()
        print "END sum matrices calculation, elapsed time: %0.4f s" % (end - start)

        tr_i = matrix(train_index)
        te_i = matrix(test_index)

        train_grams=[]
        test_grams=[]
        for i in range(len(grams)):
            train_grams.append(grams[i][tr_i,tr_i])
            test_grams.append(grams[i][te_i,tr_i])

        y_train = Y[train_index]

        start = time.clock()

        easy = EasyMKL(lam=l, tracenorm = True)
        easy.train(train_grams, matrix(y_train))

        end = time.clock()

        del train_grams
        
        y_test = Y[test_index]
        ranktest = np.array(easy.rank(test_grams))
        rte = roc_auc_score(np.array(y_test), ranktest)
        print "END Training, elapsed time: %0.4f s" % (end - start)

        del test_grams
        del easy

        sc.append(rte)

    scores=np.array(sc)
    return scores

# TRAIN a set of kernels and return the sum matrix
def single_kernel_train_and_sum(L, train_index, target_array, folds, rs, matrix_files, shape, outer_train_index =None):
    y_train = target_array[train_index]

    for mf in mfiles:
        gram = matrix(load_svmlight_file(mf, shape, zero_based=True)[0].todense())
        if outer_train_index is not None:
            grams.append(gram[outer_train_index, outer_train_index])
        else:
            grams.append(gram)

    start = time.clock()

    tr_i = matrix(train_index)
    easy = EasyMKL(lam=L, tracenorm = True)
    easy.train([g[tr_i, tr_i] for g in grams], matrix(y_train))

    end = time.clock()
    print "END Single Training, elapsed time: %0.4f s" % (end - start)

    # reload matrices to sum them again with the weights
    start = time.clock()
    if outer_train_index is not None:
        sum_kernel = matrix(0.0, (len(outer_train_index), len(outer_train_index)))
    else:
        sum_kernel = matrix(0.0, (shape, shape))
    for w, gram in zip(easy.weights, grams):
        sum_kernel += gram * w

    end = time.clock()
    print "Kernels weighed and summed, elapsed time:", (end-start)

    del grams
    del easy
    return sum_kernel

