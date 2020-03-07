#!/usr/bin/env python3

# Based on pseudo code in the documentation of the ALF project.

import numpy as np
from dqmc_core import Hubbard, Global, HS, make_gr, sweep
from dqmc_meas import *
from dqmc_lattice import hopping_matrix
import sys


class DQMC():
    """
        Determinantal QMC object: Set parameters,
        run a QMC simulation and output the results. 

        Parameters:
        -----------
        If debug==True, the random number generator is initialized 
        with a fixed seed number to make the runs reproducible. 

        `suffix`: string,  is used to label output files in a parallel MPI run. 
    """

    def __init__(self, lattice_type, NN, Uint, mu, beta, dtau, \
        K_adj=None, output_GF=True, suffix="", debug=False):

        self.debug = debug
        if (self.debug):
            np.random.seed(seed=12345)

        # Set the lattice.
        if not K_adj:
            assert(lattice_type == 'chain')
            K_adj = hopping_matrix(lattice_type='chain', NN=NN, pbc=True)
        else:
            K_adj = np.array(K_adj)
            assert(len(K_adj.shape) == 2)
            assert(K_adj.shape[0] == K_adj.shape[1])
            assert(
                K_adj.shape[0] == NN), "Dimensions of K_adj[:,:] do not match with NN. \
                    Please check the input file."
            assert(lattice_type == "from_file"), "If you wish to specify the lattice by \
                reading the adjacency matrix from file, the `lattice_type` needs to be 'from_file'."
            
            # Check that the hopping matrix is real and symmetric. 
            assert(np.all(K_adj.imag==0))
            assert(np.allclose(K_adj.transpose(), K_adj)), 'Hopping matrix is not symmetric.'

        # Set the Hamiltonian.
        self.Hub = Hubbard(Nsites=NN, K_adj=K_adj, Uint=Uint,
                           mu=[mu, mu], beta=beta, dtau=dtau)
        self.Hub.write_simparams('simparams_model.dat')                           

        self.output_GF = output_GF
        self.suffix = suffix
        if (self.output_GF):
            self.ofname = init_output_GreenF(
                self.Hub, outfilename="GreenF", suffix=self.suffix)

        # Global data structure: HS spins, Green's functions
        # Read in the HS configuration or generate it randomly.
        self.G = Global(Nsites=NN, Ltrot=self.Hub.Ntau, init='hot')
        # Initialize parameters of the HS transformation
        self.HSparams = HS(dtau=self.Hub.dtau, Uint=self.Hub.Uint,
                           mu=self.Hub.mu, Nspecies=self.Hub.Nspecies)

        # Compute Green's function for spin up and down from scratch
        # for the first time at l=0.
        weight, sign = make_gr(
            l=0, Hub=self.Hub, G=self.G, HSparams=self.HSparams)
        # initialize the sign of the current configuration
        self.G.sign = sign

        # Fill the UDV stack by one downward sweep (missing).

    def run_MC(self, ntherm=100, nbin=10, nsweep=10):
        """
            Run MC simulation. 
        """

        # thermalization
        for itm in np.arange(ntherm):
            sweep(self.Hub, self.G, self.HSparams, iscratch=2)

        # init observables
        density_bin = []
        magnetization_bin = []
        ene_bin = []
        sign_bin = []

        # =================
        # Loop over bins
        # =================
        # timeseries_fh = open('TS.dat', 'w')
        for nbc in np.arange(nbin):
            # print("nbc=", nbc, file=sys.stderr)
            density = 0.0
            magnetization = 0.0
            ene = 0.0
            sign = 0.0
            # Loop over sweeps
            for nsw in np.arange(nsweep):
                sweep(self.Hub, self.G, self.HSparams, iscratch=2)

                # measure at l=0
                assert(self.G.current_tau == 0)
                sign += self.G.sign
                d_t = meas_density(
                    self.G.gr_up, self.G.gr_dn, self.Hub, self.G)
                density += d_t
                m_t = meas_magnetization(
                    self.G.gr_up, self.G.gr_dn, self.Hub, self.G)
                magnetization += m_t
                e_t = meas_energy(self.G.gr_up, self.G.gr_dn,
                                  self.Hub.K_adj, self.Hub, self.G)
                ene += e_t
                # timeseries_fh.write("%16.8f %16.8f %16.8f \n" % (e_t, m_t, d_t))

                if (self.output_GF):
                    # Output Green's function for spin up and down.
                    output_GreenF(self.G, self.Hub, outfilename=self.ofname)

            density_bin.append(density / nsweep)
            magnetization_bin.append(magnetization / nsweep)
            ene_bin.append(ene / nsweep)
            sign_bin.append(sign / nsweep)

        # timeseries_fh.close()
        density_bin = np.array(density_bin)
        magnetization_bin = np.array(magnetization_bin)
        ene_bin = np.array(ene_bin)
        sign_bin = np.array(sign_bin)

        sign_av = np.average(sign_bin)
        # take care of sign problem in a generic way
        density_av = np.average(density_bin / sign_bin)
        density_err = np.std(density_bin / sign_bin) / np.sqrt(nbin)
        magnetization_av = np.average(magnetization_bin / sign_bin)
        magnetization_err = np.std(
            magnetization_bin / sign_bin) / np.sqrt(nbin)
        ene_av = np.average(ene_bin / sign_bin)
        ene_err = np.std(ene_bin / sign_bin) / np.sqrt(nbin)
        # print(1.0/self.beta, self.mu, ene_av, ene_err, density_av,
        #   density_err, magnetization_av, magnetization_err, sign_av)


if __name__ == "__main__":

    for temp in np.linspace(0.25, 0.25, 1):
        simu = DQMC(lattice_type='chain', NN=4,
                    Uint=4.0, mu=2.0, beta=1.0/temp, dtau=0.125, K_adj=None, output_GF=True)
        simu.run_MC(ntherm=2, nbin=1, nsweep=2)
        del simu
