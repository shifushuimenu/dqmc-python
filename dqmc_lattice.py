#!/usr/bin/env python3 
""" Generation of lattice structures encoded in the hopping matrix. """
import numpy as np

def hopping_matrix(lattice_type='chain', NN=4, pbc=True):
    """ 
        Returns `K_adj`, the adjacency matrix for the 
        lattice structure defined by `lattice_type`.
    """
    if (lattice_type == 'chain'):
        hh = -1*np.ones(NN-1)
        K_adj = np.diag(hh, k=1) + np.diag(hh, k=-1)
        if (pbc):
            K_adj[0, NN-1] = K_adj[NN-1, 0] = -1
    else:
        raise NotImplementedError 

    return K_adj