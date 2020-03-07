#!/usr/bin/env python3
"""Routines for numerically stable calculation of large chains of matrix products."""

# Relative and absolute tolerance for accuracy 
# of Green's function computation. 
RTOL=1e-3
ATOL=1e-4

# def stablize_0b_svd(n):
#     # Spin up
#     Umat1[:,:] = Ustack_up[:,:,n-1]
#     Dved1[:,:] = Dstack_up[:,n-1]
#     Vmat1[:,:] = Vstack_up[:,:,n-1]

#     # Spin down    
#     Umat1[:,:] = Ustack_dn[:,:,n-1]
#     Dved1[:,:] = Dstack_dn[:,n-1]
#     Vmat1[:,:] = Vstack_dn[:,:,n-1]    
#     pass

# def stabilize_b0_svd(n):        
#     pass