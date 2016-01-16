import sys, os
import time
import numpy as np
from svmlight_loader import dump_svmlight_file
from skgraph.datasets import load_graph_datasets
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL

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

print "Saving Gram matrices..."
for idx, kernel_matrix in enumerate(grams):
    print kernel_matrix.shape
    output = outfile+".idx"+str(idx)+".svmlight"
    dump_svmlight_file(kernel_matrix, target_array, output, True)
print '--- done'

