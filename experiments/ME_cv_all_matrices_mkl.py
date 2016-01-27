import sys, os
#sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from cvxopt import matrix, mul
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from ME_innerCV_easyMKL import calculate_inner_AUC_kfold

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile seed shape matrix_files*")

Lambda = float(sys.argv[1])
output = sys.argv[2]
seed = int(sys.argv[3])
ncols = int(sys.argv[4])
mfiles = sys.argv[5:len(sys.argv)]
nfolds = 10

def calculate_outer_AUC_kfold(matrix_files, shape, L, rs, folds, outfile):
    sc = []

    f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')

    km, target_array = load_svmlight_file(matrix_files[0], shape, zero_based=True)
    del km
    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

    f.write("Total examples "+str(shape)+"\n")
    f.write("CV\t\t test_roc_score\n")

    # OUTER K-FCV
    for train_index, test_index in kf:
        easy = EasyMKL(lam=L, tracenorm = True)

        tr_i = matrix(train_index)
        te_i = matrix(test_index)
        y_train = target_array[train_index]
        y_test = target_array[test_index]

        # COMPUTE INNER K-FOLD
        print "Computing inner "+str(folds)+"FCV..."
        inner_scores = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape, tr_index=train_index)
        print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

        f.write(str(inner_scores.mean())+"\t")

        # load matrices to sum them with ntrace norm
        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        for mf in matrix_files:
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            trainmat = matrix(km.todense())[tr_i, tr_i]

            ntrace = easy.traceN(trainmat)
            easy.traces.append(ntrace)

            if ntrace > 0.:
                train_gram += (trainmat / ntrace)
            else:
                train_gram += trainmat

        # STEP 1: preliminar training with easyMKL
        print "Outer training..."
        start = time.clock()

        easy.train(train_gram, matrix(y_train))

        # STEP 2: Weights evaluation + sum kernel with weights calculation:
        yg =  mul(easy.gamma.T, easy.labels.T)
        easy.weights = []

        # on-the-fly load and calculations
        for mf, nt in zip(matrix_files, easy.traces):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            if nt > 0:
                kermat = matrix(km.todense())[tr_i, tr_i] / nt
            else:
                kermat = matrix(km.todense())[tr_i, tr_i]
            b = yg*kermat*yg.T
            easy.weights.append(b[0])
            
        norm2 = sum(easy.weights)
        easy.weights = [w / norm2 for w in easy.weights]

        for idx,val in enumerate(easy.traces):
            if val > 0.:
                easy.weights[idx] = easy.weights[idx] / val        

        test_grams = []
        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        # reload matrices to sum them again with the weights
        for w, mf in zip(easy.weights, matrix_files):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            kermat = matrix(km.todense())
            train_gram += (kermat[tr_i, tr_i]) * w
            test_grams.append(kermat[te_i, tr_i])

        # STEP 3 final training with easyMKL with weights incorporated
        easy.train2(train_gram, matrix(y_train))

        test_gram = matrix(0.0, (len(test_index), len(train_index)))
        for w, te_g in zip(easy.weights, test_grams):
            test_gram += te_g * w

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)

        # predict on test examples
        ranktest = np.array(easy.rank(test_gram))
        rte = roc_auc_score(np.array(y_test), ranktest)

        del test_grams
        del train_gram
        del test_gram
        del easy

        f.write(str(rte)+"\t")
        f.write(str(inner_scores.std())+"\n")

        sc.append(rte)

    f.close()

    scores=np.array(sc)

    print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

def calculate_single_AUC_kfold_and_sum(L, train_index, test_index, target_array, folds, rs, matrix_files, shape):
        easy = EasyMKL(lam=L, tracenorm = True)

        tr_i = matrix(train_index)
        y_train = target_array[train_index]

        # COMPUTE INNER K-FOLD
        print "Computing inner "+str(folds)+"FCV..."
        inner_scores = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape)
        print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

        # load matrices to sum them with ntrace norm
        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        for mf in matrix_files:
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            trainmat = matrix(km.todense())[tr_i, tr_i]

            ntrace = easy.traceN(trainmat)
            easy.traces.append(ntrace)

            if ntrace > 0.:
                train_gram += trainmat / ntrace
            else:
                train_gram += trainmat

        # STEP 1: preliminar training with easyMKL
        print "Outer training..."
        start = time.clock()

        easy.train(train_gram, matrix(y_train))

        del train_gram

        # STEP 2: Weights evaluation + sum kernel with weights calculation:
        yg =  mul(easy.gamma.T, easy.labels.T)
        easy.weights = []

        # on-the-fly load and calculations
        for mf, nt in zip(matrix_files, easy.traces):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            if nt > 0.:
                kermat = matrix(km.todense())[tr_i, tr_i] / nt
            else:
                kermat = matrix(km.todense())[tr_i, tr_i]
            b = yg*kermat*yg.T
            easy.weights.append(b[0])
            
        norm2 = sum(easy.weights)
        easy.weights = [w / norm2 for w in easy.weights]

        for idx,val in enumerate(easy.traces):
            if val > 0.:
                easy.weights[idx] = easy.weights[idx] / val        

        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        # reload matrices to sum them again with the weights
        for w, mf in zip(easy.weights, matrix_files):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            kermat = matrix(km.todense())
            train_gram += kermat[tr_i, tr_i] * w

        # STEP 3 final training with easyMKL with weights incorporated
        easy.train2(train_gram, matrix(y_train))

        del train_gram

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)

        res_indices = np.concatenate((train_index, test_index))
        res_indices.sort()
        r_i = matrix(res_indices)
        sum_kernel = matrix(0.0, (len(res_indices), len(res_indices)))
        for w, mf in zip(easy.weights, matrix_files):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            kermat = matrix(km.todense())
            sum_kernel += kermat[r_i,r_i] * w

        return sum_kernel

start = time.clock()

calculate_outer_AUC_kfold(mfiles, ncols, Lambda, seed, nfolds, output)

end = time.clock()
print "END seed:", rs, "; elapsed time: %0.4f s" % (end - start)
