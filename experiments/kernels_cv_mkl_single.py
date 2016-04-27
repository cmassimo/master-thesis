import sys, os, shutil
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from itertools import chain
from cvxopt import matrix
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from ME_innerCV_easyMKL import calculate_inner_AUC_kfold

from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile seeds kernels dataset")

Lambda = float(sys.argv[1])
output = sys.argv[2]
seeds = map(lambda x: int(x), sys.argv[3].split(',')) 
kernels = sys.argv[4].split(',')
dataset = sys.argv[5]

radiuses = range(1,11)
nfolds = 10

def calculate_outer_AUC_kfold(grams, target_array, L, rs, folds, outfile):
    f = open(outfile, 'w')

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

start = time.clock()
#compute matrices on the fly :O
print "Generating orthogonal matrices"

grams = []
for r in radiuses:
    k = ODDSTincGraphKernel(r=r, l=1, normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels)
    grams += [matrix(np.array(g, dtype='float64')) for g in k.computeKernelMatrixTrain(ds.graphs)]
    for key in kernels:
        for idx in range(r+1):
            print key+".r."+str(r)+".depth."+str(idx)
end = time.clock()
print "Matrices computed in:", str(end - start)

for rs in seeds:
    fname = output+".seed"+str(rs)+".L"+str(Lambda)
    if not os.path.isfile(fname):
        calculate_outer_AUC_kfold(grams, target_array, Lambda, rs, nfolds, fname)

    print "END seed", rs

print "END"
