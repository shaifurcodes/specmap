from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from scipy.spatial import cKDTree as KDTree


@cython.wraparound (False)
@cython.boundscheck(False)
cpdef np.ndarray[np.float64_t, ndim=1] idw( np.ndarray [np.float64_t, ndim=2] X, np.ndarray [np.float64_t, ndim=1] z, np.ndarray[np.float64_t, ndim=2] q, int leafsize=16, int nnear=6, float eps=0, float p=2.0):
        # nnear nearest neighbours of each query point --

    tree = KDTree( X, leafsize=leafsize )  # build the tree

    distances, ix = tree.query( q, k=nnear, eps=eps )
    cdef np.ndarray[np.float64_t, ndim=1] interpol = np.zeros( (len(distances)), dtype=np.float64 )
    cdef int jinterpol = 0
    cdef np.float64_t wz = 0.0
    cdef np.ndarray[np.float64_t, ndim=1] w
    cdef np.float64_t wsum = 0.0
    for dist, ix in zip( distances, ix ):
        if nnear == 1:
            wz = z[ix]
        elif dist[0] < 1e-10:
            wz = z[ix[0]]
        else:  # weight z s by 1/dist --
            w = 1 / dist**p
            wsum = np.sum(w)
            np.divide(w, wsum, out=w)
            wz = np.dot( w, z[ix] )
        interpol[jinterpol] = wz
        jinterpol += 1
    return interpol
#...............................................................................