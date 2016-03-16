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
    sys.exit("python cv_all_matrices_mkl.py Ls outfile shape seeds matrix_files*")

Lambdas = map(lambda x: float(x), sys.argv[1].split(','))
output = sys.argv[2]
ncols = int(sys.argv[3])
seeds = map(lambda x: int(x), sys.argv[4].split(',')) 
mfiles = sys.argv[5:len(sys.argv)]
nfolds = 10

def calculate_outer_AUC_kfold(grams, target_array, L, rs, folds, outfile):
    f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')

    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

    f.write("Total examples "+str(len(target_array))+"\n")
    f.write("CV\t\t test_roc_score\n")

    # OUTER K-FCV
    for train_index, test_index in kf:
        y_train = target_array[train_index]

        # COMPUTE INNER K-FOLD
        inner_scores = calculate_inner_AUC_kfold(grams, y_train, l=L, rs=rs, folds=folds, tr_index=train_index)

        f.write(str(inner_scores.mean())+"\t")

        tr_i = matrix(train_index)
        easy = EasyMKL(lam=L, tracenorm = True)
        easy.train([g[tr_i, tr_i] for g in grams], matrix(y_train))

        # predict on test examples
        te_i = matrix(test_index)
        y_test = target_array[test_index]
        ranktest = np.array(easy.rank([g[te_i, tr_i] for g in grams]))
        rte = roc_auc_score(np.array(y_test), ranktest)

        f.write(str(rte)+"\t")
        f.write(str(inner_scores.std())+"\t")
        f.write(",".join(map(lambda x: str(x), easy.weights)) + "\n")

        del easy

    f.close()

start = time.clock()
#load matrices the usual way without processing
grams = [matrix(load_svmlight_file(mf, ncols, zero_based=True)[0].todense()) for mf in mfiles]
end = time.clock()
print "Matrices loaded in:", str(end - start)

target_array = load_svmlight_file(mfiles[0], ncols, zero_based=True)[1]

for rs in seeds:
    for l in Lambdas:
        calculate_outer_AUC_kfold(grams, target_array, l, rs, nfolds, output)

print "END"
