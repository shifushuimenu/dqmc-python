#!/usr/bin/env python3

# Based on pseudo code in the documentation of the ALF project.

import numpy as np
from dqmc_core import Hubbard, Global, HS, make_gr, sweep

NN = 4

# Set the lattice and the Hamiltonian.
hh = -1*np.ones(NN-1)
K_adj = np.diag(hh, k=1) + np.diag(hh, k=-1)
K_adj[0, NN-1] = K_adj[NN-1, 0] = -1  # pbc
Hub = Hubbard(Nsites=NN, K_adj=K_adj, Uint=1.0, mu=[0.5, 0.5], beta=2.00, dtau=0.125)

# Global data structure: HS spins, Green's functions
# Read in the HS configuration or generate it randomly.
G = Global(Nsites=NN, Ltrot=Hub.Ntau, init='hot')
# Initialize parameters of the HS transformation 
HSparams = HS(dtau=Hub.dtau, Uint=Hub.Uint, mu=Hub.mu, Nspecies=Hub.Nspecies)

# Compute Green's function for spin up and down from scratch. 
make_gr(l=0, Hub=Hub, G=G, HSparams=HSparams)

# Fill the UDV stack by one downward sweep (missing).


nbin = 10
nsweep = 100

# Loop over bins
for nbc in np.arange(nbin):

    # Loop over sweeps
    for nsw in np.arange(nsweep):
        print("nbc, nsw=", nbc, nsw)
        sweep(Hub, G, HSparams, iscratch=8)

    print("half filling ? homogeneous density ")
    print(G.gr_up.diagonal())