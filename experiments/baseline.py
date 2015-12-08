# MKL with all the new approaches vanilla (needs precomputed kernel matrices)

# decompose each kernel in N feature buckets, where N = max_radius
# ODDSTCGK
# ODDSTPCGK
# WLDDK
# WLNSK

from skgraph.kernel.ODDSTCGraphKernel import ODDSTCGraphKernel
from skgraph.kernel.ODDSTPCGraphKernel import ODDSTPCGraphKernel
from skgraph.kernel.WLDDKGraphKernel import WLDDKGraphKernel
from skgraph.kernel.WLNSKGraphKernel import WLNSKGraphKernel
from skgraph.kernel.EasyMKL.EasyMKL import EasyMKL
from sklearn.datasets import load_svmlight_file
import numpy as np

gram_files = [
        "oddstc.gram",
        "oddstpc.gram",
        "wlddk.gram",
        "wlnsk.gram"
        ]

prefix = "./grams/"

kernels_and_targets = [load_svmlight_file(prefix + gfile) for gfile in gram_files]

klist = [kt[0] for kt in kernels_and_targets]
Y = kernels_and_targets[0][1]

# 10FCV setup here

l = 0.5
easy = EasyMKL(lam = l, tracenorm = True)

easy.train(klist, Y)

kernel_matrix = easy.sum_kernels(klist, easy.weights)
