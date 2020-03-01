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