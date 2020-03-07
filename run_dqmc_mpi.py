from mpi4py import MPI
import numpy as np
from dqmc_main import DQMC
import argparse

# Parse command line arguments. 
parser = argparse.ArgumentParser(
    description="Perform a determinantal QMC simulation of the Hubbard model on the square lattice (default: 2x2).")
parser.add_argument('Uint', metavar='U', type=float, 
    help="Hubbard interaction U (U>0 for repulsive interactions)")
parser.add_argument('mu', metavar='mu', type=float,
    help="Chemical potential (equal for spin up and down); mu=U/2 is half filling.")
parser.add_argument('temp', metavar='temp', type=float, 
    help="Temperature (in units of the hopping matrix elements).")
parser.add_argument('--dtau', type=float,
    help="Trotter discretization parameter; default=0.125.")
parser.add_argument('--ntherm', type=int, 
    help="Number of thermalization MCS (sweep up and down); default=400.")    
parser.add_argument('--nbin', type=int,
    help="Number of bins; default=10.")
parser.add_argument('--nsweep', type=int,
    help="Number of sweeps per bin; default=2000.")
parser.add_argument('--no_storeGF', action="store_true",
    help="Suppress storage of Green's function at time slice tau=0 after every sweep.")
parser.add_argument('--Kmatrix', metavar='filename', type=str,
    help="Read a hopping matrix from file.")    
args = parser.parse_args()    

try:
   K_adj = np.loadtxt(args.Kmatrix)
except:
    print("Cannot open file %s."%(args.Kmatrix))
    exit()


# MPI communicator
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Check that MPI is working correctly:
print("my rank is %5.5d \n" % rank)

simu = DQMC(
    lattice_type='chain', NN=4,
    Uint=args.Uint, 
    mu=args.mu, 
    beta=1.0/args.temp, 
    dtau=args.dtau if args.dtau else 0.125, 
    output_GF=not args.no_storeGF, 
    suffix="_ncpu%5.5d" % rank
            )
simu.run_MC(
    ntherm=args.ntherm if args.ntherm else 400, 
    nbin=args.nbin if args.nbin else 10, 
    nsweep=args.nsweep if args.nsweep else 2000
            )

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
