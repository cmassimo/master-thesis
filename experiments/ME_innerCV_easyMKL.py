# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:22:03 2015

Copyright 2015 Nicolo' Navarin

This file is part of scikit-learn-graph.

scikit-learn-graph is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

scikit-learn-graph is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with scikit-learn-graph.  If not, see <http://www.gnu.org/licenses/>.
"""
import time
import numpy as np
from cvxopt import matrix, mul
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file

def calculate_inner_AUC_kfold(Y, l, rs, folds, mfiles, shape, tr_index):
    out_index = matrix(tr_index)
    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    easy = EasyMKL(lam=l, tracenorm = True)
    splitfolds = [sfolds for sfolds in kf]
    initial_train_grams = [matrix(0.0, (len(s[0]), len(s[0]))) for s in splitfolds]
    all_ntraces = [[] for s in splitfolds]

    start = time.clock()

    # load matrices to sum them with ntrace norm
    for mf in mfiles:
        km, ta = load_svmlight_file(mf, shape, zero_based=True)
        mat = matrix(km.todense())[out_index,out_index]
        for i, ids in enumerate(splitfolds):
            ntraces = []
            tr_i = matrix(ids[0])
            trainmat = mat[tr_i, tr_i]

            ntrace = easy.traceN(trainmat)
            all_ntraces[i].append(ntrace)

            if ntrace > 0.:
                initial_train_grams[i] += (trainmat / ntrace)
            else:
                initial_train_grams[i] += trainmat

    end = time.clock()
    print "Matrices loaded in: " + str(end - start)

    for i, (train_index, test_index) in enumerate(splitfolds):
        easy = EasyMKL(lam=l, tracenorm = True)

        tr_i = matrix(train_index)
        te_i = matrix(test_index)
        Ytr = Y[train_index]
        Yte = Y[test_index]

        print "Inner training..."
        start = time.clock()

        train_gram = initial_train_grams[i]
        easy.traces = all_ntraces[i]

        print train_gram.size, len(Ytr)

        easy.train(train_gram, matrix(Ytr))

        print "Step 2: weights"
        # STEP 2: Weights evaluation + sum kernel with weights calculation:
        yg =  mul(easy.gamma.T, easy.labels.T)
        easy.weights = []

        # on-the-fly load and calculations
        for mf, nt in zip(mfiles, easy.traces):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            mat = matrix(km.todense())[out_index, out_index]
            if nt > 0:
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

        train_gram = matrix(0.0, (len(train_index), len(train_index)))
#        test_gram = matrix(0.0, (len(test_index), len(train_index)))
        test_grams = []
        # reload matrices to sum them again with the weights
        for w, mf in zip(easy.weights, mfiles):
            km, ta = load_svmlight_file(mf, shape, zero_based=True)
            kermat = matrix(km.todense())[out_index, out_index]
            train_gram += kermat[tr_i, tr_i] * w
            test_grams.append(kermat[te_i, tr_i])

        print "Step 3: final training"
        # STEP 3 final training with easyMKL with weights incorporated
        easy.train2(train_gram, matrix(Ytr))

        # weight test_grams with the latest computed weights
        test_gram = matrix(0.0, (len(test_index), len(train_index)))
        for w, te_g in zip(easy.weights, test_grams):
            test_gram += te_g * w

        print "--- Ranking..."
        ranktest = np.array(easy.rank(test_gram))
        rte = roc_auc_score(np.array(Yte), ranktest)

        end = time.clock()
        print "kth AUC score: ", rte

        del test_grams
        del train_gram
        del test_gram
        del easy

        print 'Inner K-fold, elapsed time:', (end-start)

        sc.append(rte)

    scores=np.array(sc)
    return scores

