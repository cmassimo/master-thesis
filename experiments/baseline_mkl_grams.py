import sys, os
import time
import numpy as np
from cvxopt import matrix
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from svmlight_loader import dump_svmlight_file

if len(sys.argv)<4:
    sys.exit("python baseline.py kernels dataset radius outfile")

kernels =  sys.argv[1].split(',')
dataset = sys.argv[2]
radius = int(sys.argv[3])
outfile = sys.argv[4]+"_".join(kernels)+".d"+dataset+".r"+str(radius)

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
k = ODDSTincGraphKernel(r=radius, l=1, normalization=True, version=1, ntype=0, nsplit=0, kernels=kernels)
grams = [np.matrix(g) for g in k.computeKernelMatricesTrain(ds.graphs)]
print '--- done'

# Preprocessing for memory efficient easyMKL
tracenorm_sum_gram = np.zeros(grams[0].shape)
for gs in grams:
    ntrace = sum([gs[i,i] for i in range(gs.shape[0])]) / gs.shape[0]
    tracenorm_sum_gram += gs / ntrace

print "Saving SUM Matrix..."
dump_svmlight_file(tracenorm_sum_gram, target_array, outfile+"_SUM_tracen.svmlight", True)
print '--- done'

print "Saving Gram matrices..."
for idx, kernel_matrix in enumerate(grams):
    output = outfile+".idx"+str(idx)+".svmlight"
    dump_svmlight_file(kernel_matrix, target_array, output, True)
#    output = open(outfile+".idx"+str(idx)+".svmlight", "w")
#    for i in xrange(len(kernel_matrix)):
#        output.write(str(ds.target[i])+" 0:"+str(i+1)+" ")
#        for j in range(len(kernel_matrix[i])):
#            output.write(str(j+1)+":"+str(kernel_matrix[i][j])+" ")
#        output.write("\n")
#    output.close()
print '--- done'

