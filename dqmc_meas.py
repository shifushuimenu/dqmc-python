#!/usr/bin/env python3
"""Equal-time observables measured via Wick's theorem. 
   (Inefficient implementation)"""

import numpy as np
from dqmc_core import Hubbard, Global


def meas_energy(gr_up, gr_dn, K_adj, Hub, G):
    """
        Energy estimator in one Hubbard-Stratonovich sample (per lattice site).
        The energy scale is the hopping matrix element.

        gr_up: Single-particle Green's function for spin up.
        gr_dn: Single-particle Green's function for spin down.
        K_adj: Adjacency matrix of the kinetic term.
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))

    OBDM_up = (np.eye(*gr_up.shape) - gr_up).transpose()
    OBDM_dn = (np.eye(*gr_dn.shape) - gr_dn).transpose()

    K = -(sum(np.where(K_adj != 0, OBDM_up, 0).flatten())
          + sum(np.where(K_adj != 0, OBDM_dn, 0).flatten()))
    # chemical potential
    K += - Hub.mu[0]*sum(OBDM_up.diagonal()) - Hub.mu[1]*sum(OBDM_dn.diagonal())
    U = Hub.Uint*sum(OBDM_up.diagonal() * OBDM_dn.diagonal())

    # take care of the sign problem generically
    return G.sign * (K + U) / Hub.Nsites

def meas_magnetization(gr_up, gr_dn, Hub, G):
    """
        Compute the absolute value of the 
        average magnetization per site: |m| = |nu_up - n_down|,
        from the current Green's function. 
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))

    abs_magn = abs(sum(1.0 - gr_up.diagonal()) - sum(1.0 - gr_dn.diagonal())) / Hub.Nsites

    # take care of the sign problem generically    
    return G.sign * abs_magn

def meas_density(gr_up, gr_dn, Hub, G):
    """
        Compute average density per site:  n = n_up + n_down,
        from the current Green's function. 
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))    

    density = (sum(1.0 - gr_up.diagonal()) + sum(1.0 - gr_dn.diagonal())) / Hub.Nsites
    
    # take care of the sign problem generically  
    return G.sign * density     


def init_output_GreenF(Hub, outfilename="GreenF", suffix=""):
    """
        suffix: e.g. 'ncpu000'
    """
    outfilename += suffix
    for s in np.arange(Hub.Nspecies):        
        outfile = outfilename+r'_up.dat' if s==0 else outfilename+r'_dn.dat'
        fh = open(outfile, 'w')
        fh.close()
    return outfilename


def output_GreenF(G, Hub, outfilename="GreenF"):
    """
        Append the i'nstantaneous (i.e. for given HS configuration) 
        single particle Green's functions for both spin species 
        to a file with the basename `outfilename`.

        Successive outputs of Green's functions are separated by two empty 
        lines. 
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))    
    assert(len(G.gr[0].shape)==2); assert(len(G.gr[1].shape)==2)
    assert(G.gr[0].shape[0] == G.gr[0].shape[1]);  assert(G.gr[1].shape[0] == G.gr[1].shape[1])
    Nsites = G.gr[0].shape[0]

    for s in np.arange(Hub.Nspecies):
        outfile = outfilename+r'_up.dat' if s==0 else outfilename+r'_dn.dat'
        with open(outfile, 'a') as fh:
            for i in np.arange(Nsites):
                fh.writelines([str(e)+"\t" for e in G.gr[s][i]]+["\n"])
            # two empty lines 
            fh.writelines("\n\n")

