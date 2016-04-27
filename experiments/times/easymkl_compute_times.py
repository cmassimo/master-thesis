import sys, os, shutil
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from cvxopt import matrix, mul
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from easymkl_innercv_times import calculate_inner_AUC_kfold

#if len(sys.argv)<4:
#    sys.exit("python compute_times.py outfile shape matrix_files*")
output = sys.argv[1]
ncols = int(sys.argv[2])
mfiles = sys.argv[3:len(sys.argv)]

#Lambdas = map(lambda x: x/10., range(11))
Lambdas = map(lambda x: x/10., range(8))
nfolds = 10
rs=42

def calculate_outer_AUC_kfold(grams, target_array, L, rs, folds):

    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

    # OUTER K-FCV
    for train_index, test_index in kf:
        y_train = target_array[train_index]

        # COMPUTE INNER K-FOLD
        calculate_inner_AUC_kfold(grams, y_train, l=L, rs=rs, folds=folds, tr_index=train_index)

        tr_i = matrix(train_index)
        easy = EasyMKL(lam=L, tracenorm = True)
        easy.train([g[tr_i, tr_i] for g in grams], matrix(y_train))

        # predict on test examples
        te_i = matrix(test_index)
        y_test = target_array[test_index]
        ranktest = np.array(easy.rank([g[te_i, tr_i] for g in grams]))
        rte = roc_auc_score(np.array(y_test), ranktest)
        del easy

grams = [matrix(load_svmlight_file(mf, ncols, zero_based=True)[0].todense()) for mf in mfiles]
target_array = load_svmlight_file(mfiles[0], ncols, zero_based=True)[1]

times_file = open(output, 'w')
times_file.write("nmat,L,time\n")

for l in Lambdas:
    start = time.clock()
    calculate_outer_AUC_kfold(grams, target_array, l, rs, nfolds)
    end = time.clock()
    times_file.write(str(len(grams))+","+str(l)+","+str(end-start)+"\n")

times_file.close()
