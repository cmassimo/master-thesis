import sys, os
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import psutil
import numpy as np
from cvxopt import spmatrix, sparse, matrix
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

p = psutil.Process(os.getpid())

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile matrix_files*")

L = float(sys.argv[1])
outfile = sys.argv[2]

# prendo solo i nomi file delle matrici gram senza estensione per passarli a load_svmlight_file
matrix_files = map(lambda f: os.path.splitext(f)[0], sys.argv[3:len(sys.argv)])

grams = []
km = None
target_array = None

start = time.clock()

for mf in matrix_files:
    km, target_array = load_svmlight_file(mf+".svmlight")
    grams.append(matrix(km.tocsc()[:,1:].todense().tolist()))
#    grams.append(km[:,1:])

end = time.clock()

print "Matrices loaded in: " + str(end - start)

print p.memory_percent()
print p.memory_info()
print "***"

# CONF VARS
folds = 10
random_states = range(42,52)

sc=[]

for rs in random_states:
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
#            train_grams.append(grams[i][train_index,:].tocsc()[:,train_index].tocsr())
#            test_grams.append(grams[i][test_index,:].tocsc()[:,train_index].tocsr())

#            index=-1    
#            for row in grams[i]:
#                index+=1    
#                if index in train_index:
#                    train_grams[i].append(np.array(row).take(train_index))
#                else:
#                    test_grams[i].append(np.array(row).take(train_index))

        y_train = target_array[train_index]
        y_test = target_array[test_index]

        # COMPUTE INNER K-FOLD
        print "Computing inner "+str(folds)+"FCV..."
        inner_scores = calculate_inner_AUC_kfold(train_grams, y_train, p, l=L, rs=rs, folds=folds)
        print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())

        f.write(str(inner_scores.mean())+"\t")

#        for i in xrange(len(train_grams)):
##            train_grams[i]=matrix(np.array(train_grams[i]))
#            coo_tmp = train_grams[i].tocoo()
#            train_grams[i] = spmatrix(coo_tmp.data.tolist(), coo_tmp.row.tolist(), coo_tmp.col.tolist(), coo_tmp.shape)
#
#        coo_tmp = None
#
#        for i in xrange(len(test_grams)):
##            test_grams[i]=matrix(np.array(test_grams[i]))
#            coo_tmp = test_grams[i].tocoo()
#            test_grams[i] = spmatrix(coo_tmp.data.tolist(), coo_tmp.row.tolist(), coo_tmp.col.tolist(), coo_tmp.shape)
#
#        coo_tmp = None

        print "Training..."
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

