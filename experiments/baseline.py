import sys, os
import time
import numpy as np
from cvxopt import matrix
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from innerCV_easyMKL import calculate_inner_AUC_kfold

if len(sys.argv)<4:
    sys.exit("python baseline.py kernel dataset radius lambda L outfile")

kernels =  sys.argv[1].split(',')
dataset = sys.argv[2]
radius = int(sys.argv[3])
lbd = float(sys.argv[4])
L = float(sys.argv[5])
outfile = sys.argv[6]+".d"+dataset+".r"+str(radius)+".l"+str(lbd)+".L"+str(L)

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

target_array = matrix(ds.target)

print "Generating orthogonal matrices"
k = ODDSTincGraphKernel(r=radius, l=lbd, normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels)
grams = k.computeKernelMatricesTrain(ds.graphs)
print '--- done'

print "Training..."
easy = EasyMKL(lam=L, tracenorm = True)
easy.train(grams, target_array)
print '--- done'

kernel_matrix = np.array(easy.sum_kernels(grams, easy.weights))

print "Saving Gram matrix..."
output = open(outfile+".svmlight", "w")
for i in xrange(len(kernel_matrix)):
    output.write(str(ds.target[i])+" 0:"+str(i+1)+" ")
    for j in range(len(kernel_matrix[i])):
        output.write(str(j+1)+":"+str(kernel_matrix[i][j])+" ")
    output.write("\n")
output.close()
print '--- done'

