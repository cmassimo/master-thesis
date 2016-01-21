import sys, os
import time
import numpy as np
from cvxopt import matrix
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL_orig import EasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile seed shape matrix_files*")

L = float(sys.argv[1])
outfile = sys.argv[2]
rs = int(sys.argv[3])
shape = int(sys.argv[4])

# prendo solo i nomi file delle matrici gram senza estensione per passarli a load_svmlight_file
matrix_files = map(lambda f: os.path.splitext(f)[0], sys.argv[5:len(sys.argv)])

grams = []
start = time.clock()

for mf in matrix_files:
    km, target_array = load_svmlight_file(mf+".svmlight", shape)
    grams.append(matrix(km.todense()))

end = time.clock()

print "Matrices loaded in: " + str(end - start)

# CONF VARS
folds = 10
random_states = [42]#range(42,52)

sc=[]

#for rs in random_states:
f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')

kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

f.write("Total examples "+str(km.shape[0])+"\n")
f.write("CV\t\t test_roc_score\n")

# OUTER K-FCV
for train_index, test_index in kf:
    train_grams=[]
    test_grams=[]
    tr_i = matrix(train_index)
    te_i = matrix(test_index)

    for i in range(len(grams)):
        train_grams.append(grams[i][tr_i,tr_i])
        test_grams.append(grams[i][te_i,tr_i])

    y_train = target_array[train_index]
    y_test = target_array[test_index]

    # COMPUTE INNER K-FOLD
    print "Computing inner "+str(folds)+"FCV..."
    inner_scores = calculate_inner_AUC_kfold(train_grams, y_train, l=L, rs=rs, folds=folds)
    print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

    f.write(str(inner_scores.mean())+"\t")

    print "Outer training..."
    start = time.clock()

    easy = EasyMKL(lam=L, tracenorm = True)
    easy.train(train_grams, matrix(y_train))

    end = time.clock()
    print "END Training, elapsed time: %0.4f s" % (end - start)
    del train_grams
    
    # predict on test examples
    ranktest = np.array(easy.rank(test_grams))
    del test_grams
    del easy
    rte = roc_auc_score(np.array(y_test), ranktest)

    f.write(str(rte)+"\t")
    
    f.write(str(inner_scores.std())+"\n")

    sc.append(rte)

f.close()

scores=np.array(sc)

print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

