from mpi4py import MPI
import numpy as np

from dqmc_main import DQMC

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# parameters
temp = 0.25
U = 4.0

simu = DQMC(lattice_type='chain', NN=4,
            Uint=U, mu=U/2.0, beta=1.0/temp, dtau=0.125, output_GF=True, suffix="_ncpu%d" % rank)
simu.run_MC(ntherm=10, nbin=10, nsweep=500)

# if rank == 0:
#     A_red = np.zeros((4,1)).flatten()
#     A = np.random.random((4,1)).flatten()
# else:
#     A = np.random.random((4,1)).flatten()

# if rank == 0:
#     comm.Reduce([A, MPI.DOUBLE], A_red, op=MPI.SUM, root=0)
#     A = A_red
# else:
#     comm.Reduce([A, MPI.DOUBLE], None, op=MPI.SUM, root=0)

# print("My rank is %d"%rank, "A=", A)
