import sys
from svmlight_loader import load_svmlight_file
import numpy as np
from cvxopt import matrix, mul
from cvxopt.lapack import syev

shape = int(sys.argv[1])
matrix_files = sys.argv[2:len(sys.argv)]

for mf in matrix_files:
    print mf, ":"
    km, ta = load_svmlight_file(mf, shape, zero_based=True)

    w = matrix(0., (1, shape))
    mat = matrix(km.todense())

    syev(mat, w)

    print "Trace:", np.array(mat).trace()

    negs = [ww for ww in w if ww < float(-1e-6)]
    print "Sum eigs:", sum(w)

    print w

    if len(negs) > 0:
        print "# negs:", len(negs), "M:", max(negs), 'm:', min(negs)
    else:
        print 'all eigenvalues >= 0 yay'

    print '----------------------'

