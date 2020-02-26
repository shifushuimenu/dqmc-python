#!/usr/bin/env python3
"""Core routines of the finite temperature (BSS) determinantal quantum Monte Carlo algorithm."""


# TODO:
# - solve problem of global variables
# - wrapping before or after updating a time slice ?
# - possible inconsistencies between duplicate variables:
#        Nsites, Ltrot, dtau
#   are shared between the classes HS, Hubbard and Global
# - resolve gr_up, gr_dn, gr[:,:,:] 


import numpy as np
from scipy import linalg


class HS():
    """
        Parameters of the single-particle Hamiltonian after 
        Hubbard-Stratonovich transformation. 
    """

    def __init__(self, dtau, Uint, mu, Nspecies):
        assert(len(mu) == Nspecies); mu=np.array(mu)
        if (Uint >= 0):
            # Auxiliary field couples to the spin.
            self.alpha = np.arccosh(np.exp(0.5*dtau*abs(Uint)))
            # bb[0]=+1, bb[-1]=-1
            self.bb = np.array([+1, -1])
            self.dtau = dtau
            self.cc = np.zeros(Nspecies, dtype=np.float32)
            self.cc[:] = dtau*(mu[:] - abs(Uint)/2)

            self.gamma = np.zeros((Nspecies, 3))
            self.gamma[:, :] = np.nan  # init with invalid values
            for spin in (1, -1):      # a[-1] is the last element of a[:]
                for s in np.arange(Nspecies):
                    self.gamma[s, spin] = np.exp(-2 *
                                                 self.alpha * self.bb[s]*spin) - 1.0
        else:
            raise NotImplementedError('Negative U Hubbard model not implemented yet.')


class Hubbard():
    """
        Global parameters.

        Energy scales are in units of the hopping matrix element. 
        The kinetic terms is assumed to be the same for both spin species. 
    """

    def __init__(self, Nsites, K_adj, Uint, mu, beta, dtau=0.02):
        self.Nsites = Nsites
        assert(K_adj.shape[0] == K_adj.shape[1])
        assert(K_adj.shape[0] == Nsites)
        assert(len(mu) == 2)
        self.K_adj = K_adj
        self.Uint = Uint
        self.mu = mu[:]
        self.beta = beta
        self.Ntau = int(beta/dtau)
        # adjust dt so that beta is precisely the chosen one
        self.dtau = self.beta/self.Ntau
        self.Nspecies = 2
        self.expmdtK = np.zeros((self.Nsites, self.Nsites))
        self.exppdtK = np.zeros((self.Nsites, self.Nsites))

        self.__make_expdtK()

    def __make_expdtK(self):
        D, U = linalg.eigh(self.K_adj)
        self.expmdtK[:, :] = np.matmul(U, np.matmul(
            np.diag(np.exp(-self.dtau*D)), U.transpose()))
        self.exppdtK[:, :] = np.matmul(U, np.matmul(
            np.diag(np.exp(+self.dtau*D)), U.transpose()))


class Global():
    """
        Global data structures for one set of parameters.
        Use different instances when parallelizing. 
    """

    def __init__(self, Nsites, Ltrot, init='hot'):
        self.Nsites = Nsites  # duplicate var (see Hamiltonian())
        self.Ltrot = Ltrot  # duplicate var
        if (init == 'hot'):
            self.HS_init_random()
        # single particle Green's function at current time slice
        # for spin up and spin down
        self.gr_up = np.zeros((self.Nsites, self.Nsites))
        self.gr_dn = np.zeros((self.Nsites, self.Nsites))
        self.gr = np.zeros((2,self.Nsites, self.Nsites)) # replace: Nspecies=2
        self.current_tau = np.nan

    def HS_init_random(self):
        self.HS_spins = np.array([+1 if s > 0 else -1 for s in np.random.randint(
            2, size=self.Nsites*self.Ltrot)]).reshape((self.Nsites, self.Ltrot))

    def HS_init_load(self):
        raise NotImplementedError


def multB_fromL(A, l, s, HS_spins, HSparams, expmdtK):
    """ 
        Multiply a dense matrix A from the left by B_l:

            A ->  B_l * A = exp(-\\Delta \\tau V^{\\sigma}(l)) * exp(-\\Delta \\tau K) * A

        s is the spin index. 
        The matrix A is changed in place. 

        The kinetic part for both spin species is assumed to be the same. 
        No checkerboard decomposition implemented.
    """
    assert(isinstance(HSparams, HS))
    HS_spins = np.array(HS_spins)

    X1 = np.matmul(expmdtK, A)
    X2 = np.diag(np.exp(HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] - HSparams.cc[s]))
    A[:,:] = np.matmul(X2, X1)


def multB_fromR(A, l, s, HS_spins, HSparams, expmdtK):
    """ 
        Multiply a dense matrix A from the right by B_l:

            A ->  A * B_l = A * exp(-\\Delta \\tau V^{\\sigma}(l)) * exp(-\\Delta \\tau K)

        s is the spin index. 
        The matrix A is changed in place. 

        The kinetic part for both spin species is assumed to be the same. 
        No checkerboard decomposition implemented.
    """
    assert(isinstance(HSparams, HS))
    HS_spins = np.array(HS_spins)
    
    X1 = np.diag(np.exp(HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] - HSparams.cc[s]))
    X2 = np.matmul(A, X1)         # IMPROVE: sparse matrix multiplication
    A[:,:] = np.matmul(X2, expmdtK)  # IMPROVE: checkerboard decomposition


def multinvB_fromR(A, l, s, HS_spins, HSparams, exppdtK):
    """
        Multiply a dense matrix A from the right by B_l^{-1}:

            A -> A * B_{l}^{-1} = A * exp(+\\Delta \\tau K) * exp(+\\Delta \\tau V^{\\sigma}(l))

        s is the spin index. 
        The matrix A is changed in place. 

        The kinetic part for both spin species is assumed to be the same. 
        No checkerboard decomposition implemented.
    """
    assert(isinstance(HSparams, HS))
    HS_spins = np.array(HS_spins)

    X1 = np.matmul(A, exppdtK)
    X2 = np.diag(np.exp(-HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] + HSparams.cc[s]))
    A[:,:] = np.matmul(X1, X2)    


def multinvB_fromL(A, l, s, HS_spins, HSparams, exppdtK):
    """
        Multiply a dense matrix A from the left by B_l^{-1}:

            A -> B_{l}^{-1} * A = exp(+\\Delta \\tau K) * exp(+\\Delta \\tau V^{\\sigma}(l)) * A 

        s is the spin index. 
        The matrix A is changed in place. 

        The kinetic part for both spin species is assumed to be the same. 
        No checkerboard decomposition implemented.
    """
    assert(isinstance(HSparams, HS))
    HS_spins = np.array(HS_spins)

    X1 = np.diag(np.exp(-HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] + HSparams.cc[s]))
    X2 = np.matmul(X1, A)          # IMPROVE: sparse matrix multiplication
    A[:,:] = np.matmul(exppdtK, X2)  # IMPROVE: checkerboard decomposition


def make_gr(l, Hub, G, HSparams, stab_type='svd', istab=8):
    """
        Compute the single-particle Green's function at time slice l
        from scratch according to:

            G(l)^{s} = ( 1 + B_{l} B_{l-1} ... B_1 B_L ... B_{l+1} )^{-1}

        l: time slice  

        Stabilize the long chain of matrix multiplications by either 
        SVD or QR decomposition.  

        It is assumed that the Green's function gr[:,:] has the right shape.
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))
    
    # unnecessary helper variable 
    gr = np.zeros((Hub.Nspecies, Hub.Nsites, Hub.Nsites))

    for s in range(Hub.Nspecies):

        lrange = np.roll(np.arange(Hub.Ntau)[::-1], l)
        A = np.eye(Hub.Nsites)
        for idx, ll in enumerate(lrange[::-1]):
            # kinetic term is assumed to be the same for both spin species 
            multB_fromL(A, ll, s, G.HS_spins, HSparams, Hub.expmdtK)
            if (idx % istab == 0):
                # # stabilization: UDV decomposition of a *column-stratified matrix* is numerically stable 
                # A = UD_term 
                # # Chains of unitary matrices can be multiplied together in a stable manner. 
                # V = V_term.dot(V) 
                pass
        # A = A.dot(V)

        gr[s] = linalg.inv(np.eye(Hub.Nsites) + A)

    G.gr_up = gr[0]
    G.gr_dn = gr[1]
    G.gr[0,:,:] = gr[0]
    G.gr[1,:,:] = gr[1]


def update_gr_Metropolis(gr, i, l, Hub, G, HSparams):
    """
        Low-rank update of the Green's function after a single-spin flip
        update at space-time site (i,l).

        gr[0:Nspecies, 0:Nsites, 0:Nsites] is the Green's function at time slice l.
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))

    ratio = np.zeros(Hub.Nspecies, dtype=np.float32)
    # determinant ratios
    R = 1.0
    for s in np.arange(Hub.Nspecies):
        ratio[s] = 1 + (1 - gr[s, i, i]) * \
            HSparams.gamma[s, G.HS_spins[i, l]]
        R *= ratio[s]

    if (R < np.random.rand()):
        # update Green's function for spin up and spin down        
        for s in np.arange(Hub.Nspecies):
            for j in np.arange(Hub.Nsites):
                for k in np.arange(Hub.Nsites):
                    if (i == k):
                        gr[s, j, k] = gr[s, j, k] \
                            - ((1.0 - gr[s, i, k]) * HSparams.gamma[s,
                                                                    G.HS_spins[i, l]] * gr[s, j, i]) / ratio[s]
                    else:
                        gr[s, j, k] = gr[s, j, k]  \
                            + (gr[s, i, k] * HSparams.gamma[s, G.HS_spins[i, l]]
                               * gr[s, j, i]) / ratio[s]

    # update the alternative variables for gr (superfluous)
    G.gr_up[:,:] = gr[0,:,:]
    G.gr_dn[:,:] = gr[1,:,:]


def wrap_north(gr, l, s, Hub, G, HSparams):
    """
        Propagate the Green's function from the current time slice l
        upward to the time slice l+1:

            G(l+1) = B_{l+1} G(l) B_{l+1}^{-1}

    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))

    multB_fromL(   gr, l+1, s, G.HS_spins, HSparams, Hub.expmdtK)
    multinvB_fromR(gr, l+1, s, G.HS_spins, HSparams, Hub.exppdtK)


def wrap_south(gr, l, s, Hub, G, HSparams):
    """
        Propagate the Green's function from the current time slice l
        downward to the time slice l-1:

            G(l-1) = B_{l}^{-1} G(l) B_{l}

    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))

    multB_fromR(gr, l, s, G.HS_spins, HSparams, Hub.expmdtK)
    multinvB_fromL(gr, l, s, G.HS_spins, HSparams, Hub.exppdtK)


def sweep_0_to_beta_init():
    """
        First sweep from 0 to beta to initialize the UDV stack. 

        Not implemented. 
    """
    pass


def sweep_0_to_beta(Hub, G, HSparams, iscratch=8):
    """
        Update the space-time lattice of auxiliary fields. 

        Initially, calculate Green's function from scratch 
        for l=0 (old, not updated time slice)  
        The l=0 time slices is updated in the downward sweep.

        For l=1,2,...,Ntau-1  do the following:
            Propagate the Green's function from time l-1 to time l,
            and compute a new estimate (using the low-rank update) of the 
            Green's function at l.        
        Stabilize every iscratch time slices (not implemented)
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))    
    # at tau=0: init UDV stack (missing)
    l = 0
    make_gr(l, Hub, G, HSparams, stab_type='svd', istab=8)
    for l in np.arange(1, Hub.Ntau, +1):  # l=1,2,...,Ntau-1
        for s in np.arange(Hub.Nspecies):
            if (s==0):
                wrap_north(G.gr_up[:,:], l-1, s, Hub, G, HSparams)
            else:
                wrap_north(G.gr_dn[:,:], l-1, s, Hub, G, HSparams)           

        for i in np.arange(Hub.Nsites):
            update_gr_Metropolis(G.gr, i, l, Hub, G, HSparams)              
        if (l % iscratch == 0):
            # Compute the propagation matrix
            # from the previous stabilization point to l
            # using the UDV stack.
            pass

        # Measure the equal-time observables.


def sweep_beta_to_0(Hub, G, HSparams, iscratch=8):
    """
        Update the space-time lattice of auxiliary fields. 
 
        For l=Ntau-2,Ntau-3,...,1,0  do the following:
            Propagate the Green's function from time l+1 to time l,
            and compute a new estimate (using the low-rank update) of the 
            Green's function at l.        
        Stabilize every iscratch time slices (not implemented)
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))
    # at tau=beta: adjust UDV stack

    for l in np.arange(Hub.Ntau-2, 0-1, -1):  # l=Ntau-2,Ntau-3,...,1,0
        for s in np.arange(Hub.Nspecies):
            wrap_south(G.gr[s,:,:], l+1, s, Hub, G, HSparams)
        for i in np.arange(Hub.Nsites):
            update_gr_Metropolis(G.gr, i, l, Hub, G, HSparams)
        if (l % iscratch == 0):
            # Compute the propagation matrix
            # from the previous stabilization point to l
            # using the UDV stack.
            pass

        # Measure the equal-time observables.


def sweep(Hub, G, HSparams, iscratch=8):
    """
        One lattice sweep consists of one sweep from 0 to beta 
        (i.e. more precisely l=1 to l=Ntau-1) and one sweep from 
        beta to 0 (i.e. l=Ntau-2 to l=0).
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))

    sweep_0_to_beta(Hub, G, HSparams, iscratch)
    sweep_beta_to_0(Hub, G, HSparams, iscratch)

    # Measure the time-displaced observables.
