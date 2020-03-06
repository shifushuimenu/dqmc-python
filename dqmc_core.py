#!/usr/bin/env python3
"""Core routines of the finite temperature (BSS) determinantal quantum Monte Carlo algorithm."""


# TODO:
# - solve problem of global variables
# - wrapping before or after updating a time slice ?
# - possible inconsistencies between duplicate variables:
#        Nsites, Ltrot, dtau
#   are shared between the classes HS, Hubbard and Global
# - resolve gr_up, gr_dn, gr[:,:,:]
# - Abstract from the details of the HS transformation by using
#   a function
#           update_HS_field(site, tau)
#           update_Greens_function(site, tau)
# - Make the wrapping operations cyclic (accept also l<0 and l>Ntau)
# - Use `@` instead of dot() or matmul(). 
# - Check redundancies in `lrange = np.roll() ...`


import numpy as np
from profilehooks import profile 
from scipy import linalg


class HS():
    """
        Parameters of the single-particle Hamiltonian after 
        Hubbard-Stratonovich transformation. 
    """

    def __init__(self, dtau, Uint, mu, Nspecies):
        assert(len(mu) == Nspecies)
        mu = np.array(mu)
        if (Uint >= 0):
            # Auxiliary field couples to the spin.
            self.alpha = np.arccosh(np.exp(0.5*dtau*abs(Uint)))
            # Different sign of the coupling for spin up and down: bb[0]=+1, bb[-1]=-1
            self.bb = np.array([+1, -1])
            self.dtau = dtau
            self.cc = np.zeros(Nspecies, dtype=np.float64)
            self.cc[:] = dtau*(mu[:] - abs(Uint)/2)

            self.gamma = np.zeros((Nspecies, 3))
            self.gamma[:, :] = np.nan  # init with invalid values
            for spin in (1, -1):       # a[-1] is the last element of a[:]
                for s in np.arange(Nspecies):
                    self.gamma[s, spin] = np.exp(-2 *
                                                 self.alpha * self.bb[s]*spin) - 1.0
        else:
            raise NotImplementedError(
                'Negative U Hubbard model not implemented yet.')


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
        elif (init == 'cold'):
            self.HS_init_cold()
        elif (init == 'load'):
            raise NotImplementedError
        # single particle Green's function at current time slice
        # for spin up and spin down
        self.gr_up = np.zeros((self.Nsites, self.Nsites))
        self.gr_dn = np.zeros((self.Nsites, self.Nsites))
        self.gr = np.zeros((2, self.Nsites, self.Nsites))  # replace: Nspecies=2
        self.current_tau = np.nan
        # sign of the weight of the current configuration
        self.sign = np.nan

    def HS_init_random(self):
        self.HS_spins = np.array([+1 if s > 0 else -1 for s in np.random.randint(
            2, size=self.Nsites*self.Ltrot)]).reshape((self.Nsites, self.Ltrot))
    def HS_init_cold(self):
        self.HS_spins = np.array([+1 for s in np.arange(self.Nsites*self.Ltrot)]).reshape((self.Nsites, self.Ltrot))

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
    
    X1 = np.matmul(expmdtK, A) # IMPROVE: checkerboard decomposition 

    # sparse matrix multiplication: Multiplication from the left with a 
    # diagonal matrix amounts to row rescaling.
    X2 = np.exp(HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] - HSparams.cc[s])
    A[:, :] = np.multiply(X2[:,None], X1)


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

    X1 = np.exp(HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] - HSparams.cc[s])
    # sparse matrix multiplication: Multiplication with a diagonal matrix from the right 
    # amounts to column rescaling. 
    X2 = np.multiply(A, X1)       
    A[:, :] = np.matmul(X2, expmdtK)  # IMPROVE: checkerboard decomposition


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

    X1 = np.matmul(A, exppdtK) # IMPROVE: checkerboard decomposition 
    # sparse matrix multiplication: Multiplication with a diagonal matrix from the right 
    # amounts to column rescaling.     
    X2 = np.exp(-HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] + HSparams.cc[s])
    A[:, :] = np.multiply(X1, X2) 


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

    X1 =  np.exp(-HSparams.alpha*HSparams.bb[s]*HS_spins[:, l] + HSparams.cc[s])
    # sparse matrix multiplication: Multiplication from the left with a 
    # diagonal matrix amounts to row rescaling.        
    X2 = np.multiply(X1[:,None], A)            
    A[:, :] = np.matmul(exppdtK, X2)  # IMPROVE: checkerboard decomposition

@profile(filename='out.prof', stdout=False)
def make_gr(l, Hub, G, HSparams, stab_type='svd', istab=8):
    """
        Compute the single-particle Green's function at time slice l
        for both spin species s \\in [0,1] from scratch according to:

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
    weight = np.zeros(Hub.Nspecies)

    for s in range(Hub.Nspecies):

        lrange = np.roll(np.arange(Hub.Ntau)[::-1], l+1)
        A = np.eye(Hub.Nsites)
        U_tmp = np.eye(Hub.Nsites)
        V_tmp = np.eye(Hub.Nsites)
        s_tmp = np.ones(Hub.Nsites)
        for idx, ll in enumerate(lrange[::-1]):
            # kinetic term is assumed to be the same for both spin species
            multB_fromL(A, ll, s, G.HS_spins, HSparams, Hub.expmdtK)
            if (idx % istab == 0):
                # stabilization: UDV decomposition of a *column-stratified matrix* is numerically stable
                A = A.dot(U_tmp).dot(np.diag(s_tmp))
                U_tmp, s_tmp, Vh = linalg.svd(A)
                # Chains of unitary matrices can be multiplied together in a stable manner.
                V_tmp = np.dot(Vh, V_tmp)
                A = np.eye(Hub.Nsites)
        # take care of the excess B_l matrices which are due to the fact
        # that 'Ntau' is not a perfect multiple of 'istab'.
        A = A.dot(U_tmp).dot(np.diag(s_tmp)).dot(V_tmp)
        U_tmp, s_tmp, V_tmp = linalg.svd(A)
        weight[s] = linalg.det(np.eye(Hub.Nsites) + A)

        # gr[s] = linalg.inv(np.eye(Hub.Nsites) + A)
        VT = V_tmp.transpose()
        UT = U_tmp.transpose()
        VUT = V_tmp.dot(U_tmp).transpose()
        gr[s] = VT.dot(linalg.inv(VUT + np.diag(s_tmp))).dot(UT)

    G.gr_up[:,:] = gr[0]
    G.gr_dn[:,:] = gr[1]
    G.gr[0, :, :] = gr[0]
    G.gr[1, :, :] = gr[1]

    G.current_tau = l
    # Return the weight of the current configuration and its sign:
    weight_tot = weight[0]*weight[1]
    G.sign = np.sign(weight_tot)

    return weight_tot, np.sign(weight_tot)


def check_gr(gr_from_scratch, gr_old, Hub):
    """
        gr_from_scratch[0:Nspecies, 0:Nsites, 0:Nsites]
        gr_old[0:Nspecies, 0:Nsites, 0:Nsites]
    """
    assert(isinstance(Hub, Hubbard))

    from dqmc_stab import RTOL, ATOL

    abort=False
    for s in np.arange(Hub.Nspecies):
        if not np.allclose(gr_from_scratch[s], gr_old[s], rtol=RTOL, atol=ATOL):
            abort=True
            print("checked gr, spins s=%d"%(s))
            print("gr_old=", gr_old[s])
            print("G.gr=", gr_from_scratch[s])
            print(np.isclose(gr_from_scratch[s], gr_old[s]))
    if(abort):
        exit()

    return True

def update_gr_lowrank(gr, i, l, Hub, G, HSparams):
    """
        Low-rank update of the single-particle Green's function
        after a single-spin flip update at space-time position 
        (space, timeslice) = (i,l), i.e. 
            G.HS_spins[i,l]  -->  -G.HS_spins[i,l].
        NOTE: The formula for the update is in terms of the *old*
              HS field configuration. Therefore HS_spins[:,:] is assumed 
              to be the *old* configuration. HS_spins[i,l] is flipped 
              explicitly *at the end* of this routine. 

        gr[0:Nspecies, 0:Nsites, 0:Nsites] is the Green's function
        for both spin species. 
        The matrix gr[:,:,:] is changed in place. 
    """
    ratio = np.zeros(Hub.Nspecies, dtype=np.float64)
    for s in np.arange(Hub.Nspecies):
        ratio[s] = 1 + (1 - gr[s, i, i]) * HSparams.gamma[s, G.HS_spins[i, l]]

    gr_old = gr.copy()  
    for s in np.arange(Hub.Nspecies):
        for j in np.arange(Hub.Nsites):
            for k in np.arange(Hub.Nsites):
                if (i == k):
                    gr[s, j, k] = gr_old[s, j, k] \
                        - ((1.0 - gr_old[s, i, k]) * HSparams.gamma[s,
                            G.HS_spins[i, l]] * gr_old[s, j, i]) / ratio[s]
                else:
                    gr[s, j, k] = gr_old[s, j, k]  \
                        + (gr_old[s, i, k] * HSparams.gamma[s, G.HS_spins[i, l]]
                            * gr_old[s, j, i]) / ratio[s]

    # update HS field configuration *after* updating the Green's function. 
    G.HS_spins[i,l] = -G.HS_spins[i,l]

@profile(filename='out.prof', stdout=False)
def Metropolis_update(gr, i, l, Hub, G, HSparams):
    """
        Update a HS field at space-time position (i,l) with Metropolis 
        or heat bath probability
        and - if the update is accepted - perform a 
        low-rank update of the Green's function.

        Parameters:
        -----------
            gr[0:Nspecies, 0:Nsites, 0:Nsites] is the Green's function 
                at time slice l.
            (i,l): space-time coordinate to be updated.

        Returns:
        --------
            updated: `True` if the single-spin flip update has been accepted,
                     `False` otherwise.
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))
    # The low-rank update is only correct if the Green's function has been
    # wrapped to the current time slice. 
    assert(G.current_tau == l)

    # determinant ratios
    ratio = np.zeros(Hub.Nspecies, dtype=np.float64)
    R = 1.0
    for s in np.arange(Hub.Nspecies):
        ratio[s] = 1 + (1 - gr[s, i, i]) * \
            HSparams.gamma[s, G.HS_spins[i, l]]
        R *= ratio[s]
    # If there is a sign problem, ...
    G.sign *= np.sign(R)
    R = abs(R)
    # heat bath acceptance probability
    # eta = R / (1.0 + R)
    # Metropolis acceptance probability
    eta = min(1.0, R)
    updated = False
    if (np.random.rand() < eta):
        updated = True
        # Update Green's function for spin up and spin down.
        # G.gr[...] is changed in place. 
        update_gr_lowrank(gr, i, l, Hub, G, HSparams)
        # NOTE: The HS field configuration is updated inside 
        # the routine update_gr_lowrank(...). Do not update 
        # the HS field configuration outside this function. 

    # update the alternative variables for gr (superfluous)
    G.gr_up[:, :] = gr[0, :, :]
    G.gr_dn[:, :] = gr[1, :, :]
    # just in case
    G.gr[:,:,:] = gr[:,:,:]

    return updated


def wrap_north(gr, l, Hub, G, HSparams):
    """
        Propagate the Green's function from the current time slice l
        upward to the time slice l+1:

            G(l+1) = B_{l+1} G(l) B_{l+1}^{-1}

        gr[0:Nspecies, 0:Nsites, 0:Nsites] is the Green's function 
        for both spin species. The matrix gr[:,:,:] is changed in place. 
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))
    assert(l<Hub.Ntau-1)
    
    for s in np.arange(Hub.Nspecies):
        multB_fromL(gr[s], l+1, s, G.HS_spins, HSparams, Hub.expmdtK)
        multinvB_fromR(gr[s], l+1, s, G.HS_spins, HSparams, Hub.exppdtK)
    
    G.current_tau = l+1


def wrap_south(gr, l, Hub, G, HSparams):
    """
        Propagate the Green's function from the current time slice l
        downward to the time slice l-1:

            G(l-1) = B_{l}^{-1} G(l) B_{l}

        gr[0:Nspecies, 0:Nsites, 0:Nsites] is the Green's function 
        for both spin species. The matrix gr[:,:,:] is changed in place.
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))
    assert(l>=0)

    for s in np.arange(Hub.Nspecies):
        multB_fromR(gr[s], l, s, G.HS_spins, HSparams, Hub.expmdtK)
        multinvB_fromL(gr[s], l, s, G.HS_spins, HSparams, Hub.exppdtK)
        
    G.current_tau = l-1


def sweep_0_to_beta_initUDV(Hub, G, HSparams, istab=8):
    """
        First sweep from 0 to beta to initialize the UDV stack. 

        Not implemented. 
    """
    assert(isinstance(Hub, Hubbard))
    assert(isinstance(G, Global))
    assert(isinstance(HSparams, HS))    

    nt = np.floor(Hub.Ntau / istab)
    Umat_up = np.zeros((Hub.Nsites, Hub.Nsites, nt))
    Vmat_up = np.zeros((Hub.Nsites, Hub.Nsites, nt))
    Dvec_up = np.zeros((Hub.Nsites, nt))

    Umat_dn = np.zeros((Hub.Nsites, Hub.Nsites, nt))
    Vmat_dn = np.zeros((Hub.Nsites, Hub.Nsites, nt))
    Dvec_dn = np.zeros((Hub.Nsites, nt))
    it = -1

    for s in range(Hub.Nspecies):

        lrange = np.roll(np.arange(Hub.Ntau)[::-1], l+1)
        A = np.eye(Hub.Nsites)
        U_tmp = np.eye(Hub.Nsites)
        V_tmp = np.eye(Hub.Nsites)
        s_tmp = np.ones(Hub.Nsites)
        for idx, ll in enumerate(lrange[::-1]):
            # kinetic term is assumed to be the same for both spin species
            multB_fromL(A, ll, s, G.HS_spins, HSparams, Hub.expmdtK)
            if (idx % istab == 0):
                it += 1
                # stabilization: UDV decomposition of a *column-stratified matrix* is numerically stable
                A = A.dot(U_tmp).dot(np.diag(s_tmp))
                U_tmp, s_tmp, Vh = linalg.svd(A)
                # Chains of unitary matrices can be multiplied together in a stable manner.
                V_tmp = np.dot(Vh, V_tmp)

                # initialize UDV stack  
                if (s == 0):
                    Umat_up[:,:,it] = U_tmp[:,:]
                    Dvec_up[:,it] = s_tmp[:]
                    Vmat_up[:,:,it] = V_tmp[:,:]
                else:
                    Umat_dn[:,:,it] = U_tmp[:,:]
                    Dvec_dn[:,it] = s_tmp[:]
                    Vmat_dn[:,:,it] = V_tmp[:,:]

                A = np.eye(Hub.Nsites)
        # take care of the excess B_l matrices which are due to the fact
        # that 'Ntau' is not a perfect multiple of 'istab'. (???)   


    return Umat_up, Dvec_up, Vmat_up, Umat_dn, Dvec_dn, Vmat_dn



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
    weight, sign = make_gr(l, Hub, G, HSparams, stab_type='svd', istab=8)

    for l in np.arange(1, Hub.Ntau, +1):  # l=1,2,...,Ntau-1
        wrap_north(G.gr, l-1, Hub, G, HSparams)
        for i in np.arange(Hub.Nsites):
            updated = Metropolis_update(G.gr, i, l, Hub, G, HSparams)          
        if (l % iscratch == 0):
            # Compute the propagation matrix
            # from the previous stabilization point to l
            # using the UDV stack. (not implemented)

            gr_old = G.gr.copy()
            # simply recompute Green's function
            weight, sign = make_gr(l, Hub, G, HSparams,
                                   stab_type='svd', istab=8)
            check_gr(G.gr, gr_old, Hub)                                   
            del gr_old

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
        wrap_south(G.gr, l+1, Hub, G, HSparams)
        for i in np.arange(Hub.Nsites):
            updated = Metropolis_update(G.gr, i, l, Hub, G, HSparams)
        if (l % iscratch == 0):
            # Compute the propagation matrix
            # from the previous stabilization point to l
            # using the UDV stack. (not implemented)

            gr_old = G.gr.copy()
            # simply recompute Green's function
            weight, sign = make_gr(l, Hub, G, HSparams,
                                   stab_type='svd', istab=8)
            check_gr(G.gr, gr_old, Hub)                                   
            del gr_old

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


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()