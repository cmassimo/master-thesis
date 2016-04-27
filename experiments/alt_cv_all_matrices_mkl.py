import sys, os, shutil
sys.path.append("/home/cmassimo/cluster_bundle/scikit-learn-graph/")
import time
import numpy as np
from cvxopt import matrix, mul
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from skgraph.kernel.EasyMKL.meEasyMKL import EasyMKL
from alt_innerCV_easyMKL import calculate_inner_AUC_kfold, compute_kernels

from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.datasets import load_graph_datasets
from itertools import product

if len(sys.argv)<4:
    sys.exit("python cv_all_matrices_mkl.py L outfile seed kernels dataset")

Lambda = float(sys.argv[1])
output = sys.argv[2]
seed = int(sys.argv[3])
kernels = sys.argv[4].split(',')
dataset = sys.argv[5]

heights = range(1,11)
lambdas = [0.1, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8]
nfolds = 10
outfile = output+".seed"+str(seed)+".L"+str(Lambda)
paramsgrid = product(heights, lambdas)

def calculate_outer_AUC_kfold(kernels, paramsgrid, dataset, L, rs, folds, outfile):
    sc = []

    f = open(outfile, 'w')

    target_array = dataset.target

    kf = cross_validation.StratifiedKFold(target_array, n_folds=folds, shuffle=True, random_state=rs)

    f.write("Total examples "+str(len(target_array))+"\n")
    f.write("CV\t\t test_roc_score\n")

    easy = EasyMKL(lam=L, tracenorm = True)
    splitfolds = [sfolds for sfolds in kf]
    initial_train_grams = [matrix(0.0, (len(s[0]), len(s[0]))) for s in splitfolds]
    all_ntraces = [[] for s in splitfolds]

    start = time.clock()

    for pt in paramsgrid:
        k = ODDSTincGraphKernel(r=pt[0], l=pt[1], normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels, buckets=False)

        for g in compute_kernels(k, ds):
            for i, ids in enumerate(splitfolds):
                tr_i = matrix(ids[0])
                trainmat = g[tr_i, tr_i]
                ntrace = easy.traceN(trainmat)
                all_ntraces[i].append(ntrace)

                if ntrace > 0.:
                    initial_train_grams[i] += (trainmat / ntrace)
                else:
                    initial_train_grams[i] += trainmat

    end = time.clock()
    print "Matrices computed in:", str(end - start)

    # OUTER K-FCV
    for i, (train_index, test_index) in enumerate(splitfolds):
        easy = EasyMKL(lam=L, tracenorm = True)

        tr_i = matrix(train_index)
        te_i = matrix(test_index)
        y_train = target_array[train_index]
        y_test = target_array[test_index]

        # COMPUTE INNER K-FOLD
        inner_scores = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape, tr_index=train_index)

        f.write(str(inner_scores.mean())+"\t")

        # STEP 1: preliminar training with easyMKL
        start = time.clock()

        train_gram = initial_train_grams[i]
        easy.traces = all_ntraces[i]

        easy.train(train_gram, matrix(y_train))

        # STEP 2: Weights evaluation + sum kernel with weights calculation:
        yg =  mul(easy.gamma.T, easy.labels.T)
        easy.weights = []

        # on-the-fly load and calculations
        for i, pt in zip(range(0,len(paramsgrid),len(kernels)), paramsgrid):
            k = ODDSTincGraphKernel(r=pt[0], l=pt[1], normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels, buckets=False)

            for j, g in enumerate(compute_kernels(k, ds)):
                if nt > 0:
                    kermat = g[tr_i, tr_i] / easy.traces[i+j]
                else:
                    kermat = g[tr_i, tr_i]
                b = yg*kermat*yg.T
                easy.weights.append(b[0])
            
        norm2 = sum(easy.weights)
        easy.weights = [w / norm2 for w in easy.weights]

        for idx,val in enumerate(easy.traces):
            if val > 0.:
                easy.weights[idx] = easy.weights[idx] / val        

        test_grams = []
        train_gram = matrix(0.0, (len(train_index), len(train_index)))
        # reload matrices to sum them again with the weights
        for i, pt in zip(range(0,len(paramsgrid),len(kernels)), paramsgrid):
            k = ODDSTincGraphKernel(r=pt[0], l=pt[1], normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels, buckets=False)

            for j, g in enumerate(compute_kernels(k, ds)):
                if nt > 0:
                    train_gram += (g[tr_i, tr_i] / easy.traces[i+j]) * easy.weights[i+j]
                else:
                    train_gram += g[tr_i, tr_i] * easy.weights[i+j]
                test_grams.append(g[te_i, tr_i])

        # STEP 3 final training with easyMKL with weights incorporated
        easy.train2(train_gram, matrix(y_train))

        test_gram = matrix(0.0, (len(test_index), len(train_index)))
        for w, te_g in zip(easy.weights, test_grams):
            test_gram += te_g * w

        end = time.clock()
        print "END Training, elapsed time: %0.4f s" % (end - start)

        # predict on test examples
        ranktest = np.array(easy.rank(test_gram))
        rte = roc_auc_score(np.array(y_test), ranktest)

        del test_grams
        del train_gram
        del test_gram

        f.write(str(rte)+"\t")
        f.write(str(inner_scores.std())+"\t")
        f.write(",".join(map(lambda x: str(x), easy.weights)) + "\n")

        del easy

        sc.append(rte)

    f.close()

    scores=np.array(sc)

    print "AUC score: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std())

#def calculate_single_AUC_kfold_and_sum(L, train_index, test_index, target_array, folds, rs, matrix_files, shape):
#        easy = EasyMKL(lam=L, tracenorm = True)
#
#        tr_i = matrix(train_index)
#        y_train = target_array[train_index]
#
#        # COMPUTE INNER K-FOLD
#        print "Computing inner "+str(folds)+"FCV..."
#        inner_scores = calculate_inner_AUC_kfold(y_train, l=L, rs=rs, folds=folds, mfiles=matrix_files, shape=shape)
#        print "Inner AUC score: %0.8f (+/- %0.8f)" % (inner_scores.mean(), inner_scores.std())
#
#        # load matrices to sum them with ntrace norm
#        train_gram = matrix(0.0, (len(train_index), len(train_index)))
#        for mf in matrix_files:
#            km, ta = load_svmlight_file(mf, shape, zero_based=True)
#            trainmat = matrix(km.todense())[tr_i, tr_i]
#
#            ntrace = easy.traceN(trainmat)
#            easy.traces.append(ntrace)
#
#            if ntrace > 0.:
#                train_gram += trainmat / ntrace
#            else:
#                train_gram += trainmat
#
#        # STEP 1: preliminar training with easyMKL
#        print "Outer training..."
#        start = time.clock()
#
#        easy.train(train_gram, matrix(y_train))
#
#        del train_gram
#
#        # STEP 2: Weights evaluation + sum kernel with weights calculation:
#        yg =  mul(easy.gamma.T, easy.labels.T)
#        easy.weights = []
#
#        # on-the-fly load and calculations
#        for mf, nt in zip(matrix_files, easy.traces):
#            km, ta = load_svmlight_file(mf, shape, zero_based=True)
#            if nt > 0.:
#                kermat = matrix(km.todense())[tr_i, tr_i] / nt
#            else:
#                kermat = matrix(km.todense())[tr_i, tr_i]
#            b = yg*kermat*yg.T
#            easy.weights.append(b[0])
#            
#        norm2 = sum(easy.weights)
#        easy.weights = [w / norm2 for w in easy.weights]
#
#        for idx,val in enumerate(easy.traces):
#            if val > 0.:
#                easy.weights[idx] = easy.weights[idx] / val        
#
#        train_gram = matrix(0.0, (len(train_index), len(train_index)))
#        # reload matrices to sum them again with the weights
#        for w, mf in zip(easy.weights, matrix_files):
#            km, ta = load_svmlight_file(mf, shape, zero_based=True)
#            kermat = matrix(km.todense())
#            train_gram += kermat[tr_i, tr_i] * w
#
#        # STEP 3 final training with easyMKL with weights incorporated
#        easy.train2(train_gram, matrix(y_train))
#
#        del train_gram
#
#        end = time.clock()
#        print "END Training, elapsed time: %0.4f s" % (end - start)
#
#        res_indices = np.concatenate((train_index, test_index))
#        res_indices.sort()
#        r_i = matrix(res_indices)
#        sum_kernel = matrix(0.0, (len(res_indices), len(res_indices)))
#        for w, mf in zip(easy.weights, matrix_files):
#            km, ta = load_svmlight_file(mf, shape, zero_based=True)
#            kermat = matrix(km.todense())
#            sum_kernel += kermat[r_i,r_i] * w
#
#        return sum_kernel

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

calculate_outer_AUC_kfold(kernels, paramsgrid, ds, Lambda, seed, nfolds, outfile)

for pt in paramsgrid:
    for key in kernels:
            print key+"-h"+str(pt[0])+"-l"+str(pt[1])

print "END seed:", seed
