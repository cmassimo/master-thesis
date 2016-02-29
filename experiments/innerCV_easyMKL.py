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
from cvxopt import matrix
from sklearn import cross_validation
import numpy as np
import time

def calculate_inner_AUC_kfold(Xs, Y, l, rs, folds):

    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    for train_index, test_index in kf:
        start = time.clock()

        Ytr=Y[train_index]    
        #print "Inner training..."
        easy = EasyMKL(lam=l, tracenorm = True)
        easy.train([matrix(Xs[i][np.ix_(train_index, train_index)]) for i in range(len(Xs))], matrix(Ytr))

        #print "--- Ranking..."
        Yte=Y[test_index]
        ranktest = np.array(easy.rank([matrix(Xs[i][np.ix_(test_index, train_index)]) for i in range(len(Xs))]))
        del easy

        rte = roc_auc_score(np.array(Yte), ranktest)

        end = time.clock()

        print 'Inner K-fold, elapsed time:', (end-start)

        sc.append(rte)

    scores=np.array(sc)
    return scores

