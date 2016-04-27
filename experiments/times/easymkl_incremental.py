import sys, os, shutil
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from cvxopt import matrix, mul
from sklearn import cross_validation
from svmlight_loader import load_svmlight_file
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.EasyMKL2 import EasyMKL
from easymkl_innercv_times import calculate_inner_AUC_kfold

from skgraph.datasets import load_graph_datasets
from skgraph.kernel.WLCGraphKernel import WLCGraphKernel
from skgraph.kernel.WLCOrthoGraphKernel import WLCOrthoGraphKernel

#if len(sys.argv)<4:
#    sys.exit("python compute_times.py outfile shape matrix_files*")

output = sys.argv[1]
ncols = int(sys.argv[2])
nmat = int(sys.argv[3])
mdir = sys.argv[4]

matrices = []
for r in range(1,11):
#for r in range(1,4):
    for l in [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]:
        # kODDSTC.r3.l1.0.mtx.svmlight
        #kWLC.r4.mtx.svmlight
        matrices.append(mdir+"/kODDSTPC.r"+str(r)+".l"+str(l)+".mtx.svmlight")
#    matrices.append(mdir+"/kWLC.r"+str(r)+".mtx.svmlight")
#dataset = sys.argv[2]
#ncols = int(sys.argv[2])
#mfiles = sys.argv[3:len(sys.argv)]

mfiles = matrices[:nmat]

Lambdas = map(lambda x: x/10., range(8))
nfolds = 10
rs=42
#iterations = [10]
#iterations = range(1,11)

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

#if dataset=="CAS":
#    print "Loading bursi(CAS) dataset"        
#    ds=load_graph_datasets.load_graphs_bursi()
#elif dataset=="GDD":
#    print "Loading GDD dataset"        
#    ds=load_graph_datasets.load_graphs_GDD()
#elif dataset=="CPDB":
#    print "Loading CPDB dataset"        
#    ds=load_graph_datasets.load_graphs_CPDB()
#elif dataset=="AIDS":
#    print "Loading AIDS dataset"        
#    ds=load_graph_datasets.load_graphs_AIDS()
#elif dataset=="NCI1":
#    print "Loading NCI1 dataset"        
#    ds=load_graph_datasets.load_graphs_NCI1()
#else:
#    print "Unknown dataset name"
#
#target_array = ds.target
#
##compute matrices on the fly :O
#
#grams = []
#for h in iterations:
#    #k1 = WLCOrthoGraphKernel(r=h, normalization=True)
#    k1 = WLCGraphKernel(r=h, normalization=True)
#    #grams += [matrix(np.array(g, dtype='float64')) for g in k1.computeKernelMatrixTrain(ds.graphs)]
#    grams.append(matrix(np.array(k1.computeKernelMatrixTrain(ds.graphs), dtype='float64')))

times_file = open(output+str(nmat), 'a')
times_file.write("nmat,L,time\n")

for l in Lambdas:
    start = time.clock()
    calculate_outer_AUC_kfold(grams, target_array, l, rs, nfolds)
    end = time.clock()
    times_file.write(str(len(grams))+","+str(l)+","+str(end-start)+"\n")

times_file.close()
