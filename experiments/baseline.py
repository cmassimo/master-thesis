# CIDM paper replication

import sys, os
import time
import numpy as np
import networkx as nx
from cvxopt import matrix
from sklearn import cross_validation
#from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

if len(sys.argv)<4:
    sys.exit("python baseline.py kernel dataset radius lambda L C outfile")

kernels =  sys.argv[1].split(',')
dataset = sys.argv[2]
radius = int(sys.argv[3])
lbd = float(sys.argv[4])
L = float(sys.argv[5])
c = float(sys.argv[6])


if dataset=="CAS":
    print "Loading bursi(CAS) dataset"        
    ds=load_graph_datasets.load_graphs_bursi()
elif dataset=="GDD":
    print "Loading GDD dataset"        
    ds=load_graph_datasets.load_graphs_GDD()
elif dataset=="CPDB":
    print "Loading CPDB dataset"        
    ds=load_graph_datasets.load_graphs_CPDB()
elif dataset=="AIDS":
    print "Loading AIDS dataset"        
    ds=load_graph_datasets.load_graphs_AIDS()
elif dataset=="NCI1":
    print "Loading NCI1 dataset"        
    ds=load_graph_datasets.load_graphs_NCI1()
else:
    print "Unknown dataset name"

target_array = ds.target

print "Generating orthogonal matrices"
k = ODDSTincGraphKernel(r=radius, l=lbd, normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels)
grams = k.computeKernelMatricesTrain(ds.graphs)
print '--- done'

sc=[]
for rs in range(42,43):
    f=open(str(sys.argv[7]+".seed"+str(rs)+".c"+str(c)),'w')

    kf = cross_validation.StratifiedKFold(target_array, n_folds=10, shuffle=True, random_state=rs)
    
    f.write("Total examples "+str(len(grams[0]))+"\n")
    f.write("CV\t\t test_acc\n")
    
    for train_index, test_index in kf:
        train_grams=[]
        test_grams=[]
        
        for i in range(len(grams)):
            train_grams.append([])
            test_grams.append([])

            index=-1    
            for row in grams[i]:
                index+=1    
                if index in train_index:
                    train_grams[i].append(np.array(row).take(train_index))
                else:
                    test_grams[i].append(np.array(row).take(train_index))

        y_train = target_array[train_index]
        y_test = target_array[test_index]

        #COMPUTE INNERKFOLD
#        print "Computing inner 10FCV..."
#        inner_scores = calculate_inner_AUC_kfold(train_grams, y_train, l=L, rs=rs, folds=3)
#        print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())
#
#        f.write(str(inner_scores.mean())+"\t")

        for i in xrange(len(train_grams)):
            train_grams[i]=matrix(np.array(train_grams[i]))

        for i in xrange(len(test_grams)):
            test_grams[i]=matrix(np.array(test_grams[i]))

        print "Training..."
        start = time.clock()

        easy = EasyMKL(lam=L, tracenorm = True)
        easy.train(train_grams, matrix(y_train))

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)
        
        # predict on test examples
        ranktest = np.array(easy.rank(test_grams))
        rte = roc_auc_score(np.array(y_test), ranktest)
    
        f.write(str(rte)+"\t")
        
#        f.write(str(inner_scores.std())+"\n")

    f.close()
scores=np.array(sc) #sc dovrebbe essere accuracy non nested sui vari random seed di 10-fold.

print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

