import sys, os, shutil
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
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
        y_train = target_array[train_index]
        easy = EasyMKL(lam=L, tracenorm = True)

        # COMPUTE INNER K-FOLD
        inner_scores, train_grams, test_grams = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape, tr_index=train_index, te_index=test_index)

        f.write(str(inner_scores.mean())+"\t")

        #TODO load all the matrices here and use FastEasyMKL
        # this way I have only 20 full matrices load.
        # maybe return the matrices from the inner so you wont load them (test memory usage)

        start = time.clock()

        easy.train(train_grams, matrix(y_train))
        del train_grams

        # predict on test examples
        y_test = target_array[test_index]
        ranktest = np.array(easy.rank(test_gram))
        rte = roc_auc_score(np.array(y_test), ranktest)
        del test_grams

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)

        f.write(str(rte)+"\t")
        f.write(str(inner_scores.std())+"\t")
        f.write(",".join(map(lambda x: str(x), easy.weights)) + "\n")

        del easy

        sc.append(rte)

    f.close()

    scores=np.array(sc)

    print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

start = time.clock()

calculate_outer_AUC_kfold(mfiles, ncols, Lambda, seed, nfolds, output)

end = time.clock()
print "END seed:", seed, ("elapsed time: %0.4f s" % (end - start))
