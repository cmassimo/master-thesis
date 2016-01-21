import sys, os
import time
import numpy as np
from cvxopt import matrix, mul
from cvxopt.lapack import syev
from svmlight_loader import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.ODDSTOrthogonalizedGraphKernel import ODDSTOrthogonalizedGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL

if len(sys.argv)<4:
    sys.exit("python baseline.py kernels dataset radius outfile")

kernels =  sys.argv[1].split(',')
dataset = sys.argv[2]
radius = int(sys.argv[3])
outfile = sys.argv[4]

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

#k = ODDSTOrthogonalizedGraphKernel(r=radius, l=1, normalization=False)
grams = [np.array(g, dtype='float64') for g in k.computeKernelMatrixTrain(ds.graphs)]
print '--- done'

print "Saving Gram matrices..."
for key in kernels:
    for idx in range(radius+1):
        kernel_matrix = grams.pop(0)

#        w = matrix(0., (1, len(target_array)))
#        print w.size, w
#        mat = matrix(kernel_matrix)
#        syev(mat, w)
#        print w.size, w
#        negs = [ww for ww in w if ww < float(-1e-6)]
#        print len(negs), negs
#        if len(negs) > 0:
#            print idx, ": # negs:", len(negs), "M:", max(negs), 'm:', min(negs)
#        else:
#            print idx, ": positive semi-definite, saving..."
        output = outfile+key+".d"+dataset+".r"+str(radius)+".depth"+str(idx)+".svmlight"
        dump_svmlight_file(kernel_matrix, target_array, output)

print '--- done'

