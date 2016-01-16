import sys, os
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
#import psutil
import numpy as np
from cvxopt import matrix, mul
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

#p = psutil.Process(os.getpid())

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile seed shape matrix_files*")

start = time.clock()

L = float(sys.argv[1])
outfile = sys.argv[2]
rs = int(sys.argv[3])
shape = int(sys.argv[4])
matrix_files = sys.argv[5:len(sys.argv)-2]

#print p.memory_percent()
#print p.memory_info()
#print "***"

folds = 10
sc = []

f = open((outfile+".seed"+str(rs)+".L"+str(L)), 'w')

km, target_array = load_svmlight_file(matrix_files[0], shape)
del km
kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

f.write("Total examples "+str(shape)+"\n")
f.write("CV\t\t test_roc_score\n")

# OUTER K-FCV
for train_index, test_index in kf:
    tr_i = matrix(train_index)
    te_i = matrix(test_index)
    y_train = target_array[train_index]
    y_test = target_array[test_index]

    # COMPUTE INNER K-FOLD
    print "Computing inner "+str(folds)+"FCV..."
    inner_scores = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape)
    print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

    f.write(str(inner_scores.mean())+"\t")

    traces = []
    # load matrices to sum them with ntrace norm
    train_gram = matrix(0.0, (len(train_index), len(train_index)))
    for mf in matrix_files:
        km, ta = load_svmlight_file(mf, shape)
        trainmat = matrix(km.todense().tolist())[tr_i, tr_i]

        ntrace = sum([trainmat[i,i] for i in xrange(trainmat.size[0])]) / trainmat.size[0]
        traces.append(ntrace)

        train_gram += trainmat / ntrace

    # STEP 1: preliminar training with easyMKL
    print "Outer training..."
    start = time.clock()

    easy = EasyMKL(lam=L, tracenorm = True)
    easy.traces = traces
    easy.train(train_gram, matrix(y_train))

    del train_gram

    # STEP 2: Weights evaluation + sum kernel with weights calculation:
    yg =  mul(easy.gamma.T, easy.labels.T)
    easy.weights = []

    # on-the-fly load and calculations
    for mf, nt in zip(matrix_files, easy.traces):
        km, ta = load_svmlight_file(mf, shape)
        kermat = matrix(km.todense())[train_index, train_index] / nt
        b = yg*kermat*yg.T
        easy.weights.append(b[0])
        
    norm2 = sum(easy.weights)
    easy.weights = [w / norm2 for w in easy.weights]

    for idx,val in enumerate(easy.traces):
        easy.weights[idx] = easy.weights[idx] / val        

    train_gram = matrix(0.0, (len(train_index), len(train_index)))
    test_gram = matrix(0.0, (len(train_index), len(test_index)))
    # reload matrices to sum them again with the weights
    for w, nt, mf in zip(easy.weights, easy.traces, matrix_files):
        km, ta = load_svmlight_file(mf, shape)
        kermat = matrix(km.todense().tolist())
        train_gram += (kermat[tr_i, tr_i] / nt ) * w
        test_gram += kermat[tr_i, te_i] * w

    # STEP 3 final training with easyMKL with weights incorporated
    easy.train2(train_gram, matrix(y_train))

    del train_gram

    end = time.clock()
    print "END Training, elapsed time: %0.4f s" % (end - start)

    # predict on test examples
    ranktest = np.array(easy.rank(test_gram))
    rte = roc_auc_score(np.array(y_test), ranktest)

    del test_gram
    del easy

    f.write(str(rte)+"\t")
    f.write(str(inner_scores.std())+"\n")

    sc.append(rte)

f.close()

scores=np.array(sc)

print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

end = time.clock()
print "END seed:", rs, "; elapsed time: %0.4f s" % (end - start)
