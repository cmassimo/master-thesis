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
from cvxopt import spmatrix, sparse, matrix
from sklearn import cross_validation
import numpy as np
import time

def calculate_inner_AUC_kfold(Xs, Y, l, rs, folds):
    #Xs is an array of kernel matrices

    sc=[]
    kf = cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=rs)

    for train_index, test_index in kf:
        train_grams=[]
        test_grams=[]

        for i in xrange(len(Xs)):
            train_grams.append(Xs[i][train_index,:].tocsc()[:,train_index].tocsr())
            test_grams.append(Xs[i][test_index,:].tocsc()[:,train_index].tocsr())
#            train_grams.append([])
#            test_grams.append([])
#
#            index=-1    
#            for row in Xs[i]:
#                index+=1
#                if index in train_index:
#                    train_grams[i].append(np.array(row).take(train_index))
#                else:
#                    test_grams[i].append(np.array(row).take(train_index))

        for i in xrange(len(train_grams)):
#            train_grams[i]=matrix(np.array(train_grams[i]))
            coo_tmp = train_grams[i].tocoo()
            train_grams[i] = spmatrix(coo_tmp.data.tolist(), coo_tmp.row.tolist(), coo_tmp.col.tolist())

        for i in xrange(len(test_grams)):
#            test_grams[i]=matrix(np.array(test_grams[i]))
            coo_tmp = test_grams[i].tocoo()
            test_grams[i] = spmatrix(coo_tmp.data.tolist(), coo_tmp.row.tolist(), coo_tmp.col.tolist())

#        print "train size: ",train_grams[0].size[0],train_grams[0].size[1]
#        print "test size: ",test_grams[0].size[0],test_grams[0].size[1]
        #print test_grams[0]

        Ytr=Y[train_index]    
        Yte=Y[test_index]

        #print "labels: ", len(Ytr), len(Yte)
        start = time.clock()

        easy = EasyMKL(lam=l, tracenorm = True)
        easy.train(train_grams,matrix(Ytr))

        ranktest = np.array(easy.rank(test_grams))
        rte = roc_auc_score(np.array(Yte),ranktest)

        end = time.clock()

        print 'Inner K-fold, elapsed time:', (end-start)
        #print 'weights of kernels:', easy.weights

        sc.append(rte)

    scores=np.array(sc)
    return scores

