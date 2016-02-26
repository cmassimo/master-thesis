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
        print "Inner training..."
        print 'Calculating single kernel sum matrices...'
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
        y_test = Y[test_index]

        start = time.clock()

        easy = FastEasyMKL(lam=l, tracenorm = True)
        easy.train(train_grams, matrix(y_train))

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)
        del train_grams
        
        # predict on test examples
        print "--- Ranking..."
        ranktest = np.array(easy.rank(test_grams))
        del test_grams
        del easy
        rte = roc_auc_score(np.array(y_test), ranktest)
        print "kth AUC score: ", rte

        sc.append(rte)

    scores=np.array(sc)
    return scores

# TRAIN a set of kernel buckets and return the sum matrix
def single_kernel_train_and_sum(L, train_index, target_array, folds, rs, matrix_files, shape, outer_train_index =None):
    easy = EasyMKL(lam=L, tracenorm = True)

    tr_i = matrix(train_index)
    y_train = target_array[train_index]

    # load matrices to sum them with ntrace norm
    train_gram = matrix(0.0, (len(train_index), len(train_index)))
    for mf in matrix_files:
        km, ta = load_svmlight_file(mf, shape, zero_based=True)
        if outer_train_index is not None:
            mat = matrix(km.todense())[outer_train_index, outer_train_index]
            trainmat = mat[tr_i, tr_i]
        else:
            trainmat = matrix(km.todense())[tr_i, tr_i]

        ntrace = easy.traceN(trainmat)
        easy.traces.append(ntrace)

        if ntrace > 0.:
            train_gram += trainmat / ntrace
        else:
            train_gram += trainmat

    # STEP 1: preliminar training with easyMKL
    start = time.clock()

    easy.train(train_gram, matrix(y_train))

    del train_gram

    # STEP 2: Weights evaluation + sum kernel with weights calculation:
    yg =  mul(easy.gamma.T, easy.labels.T)
    easy.weights = []

    # on-the-fly load and calculations
    for mf, nt in zip(matrix_files, easy.traces):
        km, ta = load_svmlight_file(mf, shape, zero_based=True)
        if outer_train_index is not None:
            mat = matrix(km.todense())[outer_train_index, outer_train_index]
        else:
            mat = matrix(km.todense())
        if nt > 0.:
            kermat = mat[tr_i, tr_i] / nt
        else:
            kermat = mat[tr_i, tr_i]
        b = yg*kermat*yg.T
        easy.weights.append(b[0])
        
    norm2 = sum(easy.weights)
    easy.weights = [w / norm2 for w in easy.weights]

    for idx,val in enumerate(easy.traces):
        if val > 0.:
            easy.weights[idx] = easy.weights[idx] / val        

#    train_gram = matrix(0.0, (len(train_index), len(train_index)))
#    for w, mf in zip(easy.weights, matrix_files):
#        km, ta = load_svmlight_file(mf, shape, zero_based=True)
#        if outer_train_index is not None:
#            kermat = matrix(km.todense())[outer_train_index, outer_train_index]
#        else:
#            kermat = matrix(km.todense())
#        train_gram += kermat[tr_i, tr_i] * w

#    # STEP 3 final training with easyMKL with weights incorporated
#    easy.train2(train_gram)

    #del train_gram

    end = time.clock()
    print "END Single Training, elapsed time: %0.4f s" % (end - start)

    # reload matrices to sum them again with the weights
    start = time.clock()
    if outer_train_index is not None:
        sum_kernel = matrix(0.0, (len(outer_train_index), len(outer_train_index)))
    else:
        sum_kernel = matrix(0.0, (shape, shape))
    for w, mf in zip(easy.weights, matrix_files):
        km, ta = load_svmlight_file(mf, shape, zero_based=True)
        if outer_train_index is not None:
            kermat = matrix(km.todense())[outer_train_index, outer_train_index]
        else:
            kermat = matrix(km.todense())
        sum_kernel += kermat * w

    end = time.clock()
    print "Kernels weighed and summed, elapsed time:", (end-start)

    del easy
    return sum_kernel

