import sys, os
import time
import numpy as np
from skgraph.kernel.ODDSTincGraphKernel import ODDSTincGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from svmlight_loader import load_svmlight_file, dump_svmlight_file

if len(sys.argv)<2:
    sys.exit("python save_sum_kernel.py shape path kernels*")

start = time.clock()
shape = int(sys.argv[1])
save_path = sys.argv[2]
matrix_files = sys.argv[3:len(sys.argv)]
tracenorm_sum_gram = np.zeros((shape,shape))

for mf in matrix_files:
    km, ta = load_svmlight_file(mf, shape+1)
    kermat = np.matrix(km.tocsc()[:,1:].todense().tolist())
    ntrace = sum([kermat[i,i] for i in range(shape)]) / shape
    tracenorm_sum_gram += kermat / ntrace

end = time.clock()

print "Summed all the matrices in:", (end-start)

start = time.clock()
dump_svmlight_file(tracenorm_sum_gram, ta, save_path+"SUM_matrix.svmlight")
end = time.clock()
print "Dumped sum matrix in:", (end-start)

