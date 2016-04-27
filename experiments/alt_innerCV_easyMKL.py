# -*- coding: utf-8 -*-

import numpy as np
from cvxopt import matrix, mul
from skgraph.kernel.EasyMKL.meEasyMKL import EasyMKL
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel

def compute_kernels(k, ds):
    return [matrix(np.array(g, dtype='float64')) for g in k.computeKernelMatrixTrain(ds.graphs)]

def calculate_inner_AUC_kfold(paramsgrid, kernels, dataset, Y, l, rs, folds, tr_index):
    out_index = matrix(tr_index)
    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    easy = EasyMKL(lam=l, tracenorm = True)
    splitfolds = [(train_index, test_index) for train_index, test_index in kf]
    initial_train_grams = [matrix(0.0, (len(s[0]), len(s[0]))) for s in splitfolds]
    all_ntraces = [[] for s in splitfolds]

    for pt in paramsgrid:
        k = ODDSTincGraphKernel(r=pt[0], l=pt[1], normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels, buckets=False)

        for g in compute_kernels(k, ds):
            mat = g[out_index,out_index]
            for i, ids in enumerate(splitfolds):
                tr_i = matrix(ids[0])
                trainmat = mat[tr_i, tr_i]
                ntrace = easy.traceN(trainmat)
                all_ntraces[i].append(ntrace)

                if ntrace > 0.:
                    initial_train_grams[i] += (trainmat / ntrace)
                else:
                    initial_train_grams[i] += trainmat

    for i, (train_index, test_index) in enumerate(splitfolds):
        easy = EasyMKL(lam=l, tracenorm = True)

        tr_i = matrix(train_index)
        te_i = matrix(test_index)
        Ytr = Y[train_index]
        Yte = Y[test_index]

        train_gram = initial_train_grams[i]
        easy.ntraces = all_ntraces[i]

        easy.train(train_gram, matrix(Ytr))

        # STEP 2: Weights evaluation + sum kernel with weights calculation:
        yg =  mul(easy.gamma.T, easy.labels.T)
        easy.weights = []

        # on-the-fly load and calculations
        for i, pt in zip(range(0,len(paramsgrid),len(kernels)), paramsgrid):
            k = ODDSTincGraphKernel(r=pt[0], l=pt[1], normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels, buckets=False)

            for j, g in enumerate(compute_kernels(k, ds)):
                mat = g[out_index,out_index]
                if nt > 0:
                    kermat = mat[tr_i, tr_i] / easy.traces[i+j]
                else:
                    kermat = mat[tr_i, tr_i]
                b = yg*kermat*yg.T
                easy.weights.append(b[0])

        norm2 = sum(easy.weights)
        easy.weights = [w / norm2 for w in easy.weights]
            
        for idx,val in enumerate(easy.traces):
            if val > 0.:
                easy.weights[idx] = easy.weights[idx] / val        

        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        test_grams = []
        # reload matrices to sum them again with the weights
        for i, pt in zip(range(0,len(paramsgrid),len(kernels)), paramsgrid):
            k = ODDSTincGraphKernel(r=pt[0], l=pt[1], normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels, buckets=False)

            for j, g in enumerate(compute_kernels(k, ds)):
                mat = g[out_index,out_index]
                if nt > 0:
                    train_gram += (mat[tr_i, tr_i] / easy.traces[i+j]) * easy.weights[i+j]
                else:
                    train_gram += mat[tr_i, tr_i] * easy.weights[i+j]
                test_grams.append(mat[te_i, tr_i])

        # STEP 3 final training with easyMKL with weights incorporated
        easy.train2(train_gram, matrix(Ytr))

        # weight test_grams with the latest computed weights
        test_gram = matrix(0.0, (len(test_index), len(train_index)))
        for w, te_g in zip(easy.weights, test_grams):
            test_gram += te_g * w

        ranktest = np.array(easy.rank(test_gram))
        rte = roc_auc_score(np.array(Yte), ranktest)

        del test_grams
        del train_gram
        del test_gram
        del easy

        sc.append(rte)

    scores=np.array(sc)
    return scores

