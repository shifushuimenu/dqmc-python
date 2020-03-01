#!/usr/bin/env python3
import numpy as np

from dqmc_core import Hubbard, Global, HS, \
     make_gr, wrap_north, wrap_south, \
     Metropolis_update, update_gr_lowrank

from dqmc_lattice import hopping_matrix

def init_testsuite():
    global K_adj, Hub, G, HSparams 
    K_adj = hopping_matrix(lattice_type='chain', NN=4, pbc=True)

    Hub = Hubbard(Nsites=4, K_adj=K_adj, Uint=6.0, mu=[
              3.0, 3.0], beta=1.0, dtau=0.0125)
    G = Global(Nsites=4, Ltrot=Hub.Ntau,  init='hot')
    HSparams = HS(dtau=Hub.dtau, Uint=Hub.Uint, mu=Hub.mu, Nspecies=2)


def test_wrapping():
    """
        Test the consistency between wrapping north and south 
        and computing the Green's function on adjacent time slices 
        from scratch. 
    """
    init_testsuite()

    weight0, sign = make_gr(0, Hub, G, HSparams)
    gr0_scratch = G.gr.copy()
    weight1, sign = make_gr(1, Hub, G, HSparams)
    gr1_scratch = G.gr.copy()
    # check cyclic property of det(1 + B_l B_{l-1} ... B_1 B_M ...B_{l+1})
    assert(np.allclose(weight0,weight1)), print(weight0, weight1)

    # check consistency between make_gr() and wrap_south()
    gr1_wrap = gr1_scratch.copy()
    wrap_south(gr1_wrap, 1, Hub, G, HSparams)
    if not np.allclose(gr0_scratch, gr1_wrap):
        print(np.isclose(gr1_wrap, gr0_scratch, atol=1e-06))

    # check consistency between make_gr and wrap_north()
    gr2_wrap = gr0_scratch.copy()
    wrap_north(gr2_wrap, 0, Hub, G, HSparams)
    if not np.allclose(gr1_scratch, gr2_wrap):
        print(np.isclose(gr2_wrap, gr1_scratch, atol=1e-06))


def test_lowrank_update1():
    """
        Test consistency between the low rank update of the Green's function 
        after a single-spin flip update and the calculation of the Green's function 
        from scratch after a single spin-flip update. 
    """
    init_testsuite()

    l=0
    make_gr(l, Hub, G, HSparams, stab_type='svd', istab=8)
    # flip a single spin 
    for i in np.arange(Hub.Nsites):
        if(Metropolis_update(G.gr, i, l, Hub, G, HSparams)):
            break
    gr1_lowrank = G.gr.copy()
    # recalculate the Green's function after the spin flip from scratch. 
    make_gr(l, Hub, G, HSparams, stab_type='svd', istab=8)
    gr1_scratch = G.gr.copy()
    C = np.isclose(gr1_lowrank, gr1_scratch)
    print("Is close:\n", C)
    print("gr1_scratch=", gr1_scratch)
    print("gr1_lowrank=", gr1_lowrank)  



if __name__ == '__main__':
    test_wrapping()        
    test_lowrank_update1()