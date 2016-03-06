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

def calculate_inner_AUC_kfold(Xs, Y, l, rs, folds, tr_index):
    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    for train_index, test_index in kf:
        tr_i = matrix(tr_index)[matrix(train_index)]
        Ytr = Y[train_index]

        start = time.clock()

        easy = EasyMKL(lam=l, tracenorm = True)
        easy.train([g[tr_i, tr_i] for g in Xs], matrix(Ytr))

        te_i = matrix(tr_index)[matrix(test_index)]
        Yte = Y[test_index]
        ranktest = np.array(easy.rank([g[te_i, tr_i] for g in Xs]))
        rte = roc_auc_score(np.array(Yte), ranktest)

        end = time.clock()

        del easy

        print 'Inner K-fold, elapsed time:', (end-start)
        sc.append(rte)

    scores=np.array(sc)

    return scores

