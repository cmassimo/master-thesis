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
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from sklearn.metrics import roc_auc_score
from cvxopt import matrix, mul
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
import numpy as np
import time

def calculate_inner_AUC_kfold(Y, l, rs, folds, mfiles, shape):
#    print process.memory_percent()
#    print process.memory_info()
#    print "---"

    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    for train_index, test_index in kf:
        traces = []
        tr_i = matrix(train_index)
        te_i = matrix(test_index)

        start = time.clock()
        # load matrices to sum them with ntrace norm
        train_gram = matrix(0., (len(train_index), len(train_index)))
        test_gram = matrix(0., (len(train_index), len(test_index)))
        for mf in mfiles:
            km, ta = load_svmlight_file(mf, shape)
            trainmat = matrix(km.todense().tolist())[tr_i, tr_i]

            ntrace = sum([trainmat[i,i] for i in xrange(trainmat.size[0])]) / trainmat.size[0]
            traces.append(ntrace)

            train_gram += trainmat / ntrace

#        print process.memory_percent()
#        print process.memory_info()
#        print "---"

        Ytr=Y[train_index]
        Yte=Y[test_index]

        end = time.clock()
        print "Matrices loaded in: " + str(end - start)

        print "Outer training..."
        start = time.clock()

        easy = EasyMKL(lam=l, tracenorm = True)
        easy.traces = traces
        easy.train(train_gram, matrix(Ytr))

        del train_gram

        # STEP 2: Weights evaluation + sum kernel with weights calculation:
        yg =  mul(easy.gamma.T, easy.labels.T)
        easy.weights = []

        # on-the-fly load and calculations
        for mf, nt in zip(mfiles, easy.traces):
            km, ta = load_svmlight_file(mf, shape)
            kermat = matrix(km.todense().tolist())[tr_i, tr_i] / nt
            b = yg*kermat*yg.T
            easy.weights.append(b[0])
            
        norm2 = sum(easy.weights)
        easy.weights = [w / norm2 for w in easy.weights]
            
        for idx,val in enumerate(easy.traces):
            easy.weights[idx] = easy.weights[idx] / val        

        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        test_gram = matrix(0.0, (len(train_index), len(test_index)))
        tr_i = matrix(train_index)
        te_i = matrix(test_index)
        # reload matrices to sum them again with the weights
        for w, nt, mf in zip(easy.weights, easy.traces, mfiles):
            km, ta = load_svmlight_file(mf, shape)
            kermat = matrix(km.todense().tolist())
            train_gram += (kermat[tr_i, tr_i] / nt ) * w
            test_gram += kermat[tr_i, te_i] * w

        # STEP 3 final training with easyMKL with weights incorporated
        easy.train2(train_gram, matrix(Ytr))

        del train_gram

        print "--- Ranking..."
        ranktest = np.array(easy.rank(test_gram))
        rte = roc_auc_score(np.array(Yte), ranktest)

        end = time.clock()

        del test_gram
        del easy

        print 'Inner K-fold, elapsed time:', (end-start)

        sc.append(rte)

    scores=np.array(sc)
    return scores

