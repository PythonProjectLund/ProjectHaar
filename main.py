# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:37:49 2017

@author: Simon
"""
from  scipy import *
from  pylab import *
import scipy.misc as sm
import numpy as np

def create_W(n):
	W = zeros([n,n])
	i = 0
	for j in range(int(n/2)):
		W[j, i] = 1/sqrt(2)
		W[j, i+1] = 1/sqrt(2)
		i += 2
	i = 0
	for j in range(int(n/2), n):
		W[j, i] = -1/sqrt(2)
		W[j, i+1] = 1/sqrt(2)
		i += 2
	return W

def preprocess_matrix(A):
    #cut away pixels to have even number of y and x pixels
    if len(A[:, 0]) % 2 != 0:
        A = A[1:, :]
    
    if len(A[0, :]) % 2 != 0:
        A = A[:, 1:]
        
    #A är en MxN matris    
    M = len(A[:, 0])
    N = len(A[0, :])
    
    return A, M, N
def compress(A):
    A, M, N = preprocess_matrix(A)
    
    W_M = create_W(M)
    W_N = create_W(N)
    
    B = dot(dot(W_M, A), W_N.T)
    
    #B1, B2, B3, B4 = 0,0,0,0#submatrices(B)

    return B#, B1, B2, B3, B4
    
def compress_no_matrices(A):
    """
    First we iterate vertically and split the
    picture into two parts
    """
    
    

def inverse_transformation(B1, B2 = None, B3 = None, B4 = None):
    """
    Takes a full compressed image B or B in parts B1, B2, B3, B4
    with diff information and returns the original matrix by
    doing the inverse transformation.
    """
    
    if not B4 == None:
        np.column_stack(B1, B2)
        np.column_stack(B3, B4)
        np.vstack(B1, B3)
        
    B = B1
    
    M = len(B[:, 0])
    N = len(B[0, :])
        
    W_M = create_W(M)
    W_N = create_W(N)
    
    A = dot(dot(W_M.T, B), W_N)
    
    return A



A = sm.imread('kvinna.jpg', True)

B = compress(A)
sm.imsave('compressed_full.jpg', B)
#sm.imsave('compressed_small.jpg', B1)

A_restored = inverse_transformation(B)

sm.imsave('decompressed.jpg', A_restored)





<<<<<<< HEAD
def submatrices(B):
    B_1 = B[:M//2,:N//2]
    B_2 = B[:M//2,N//2:]
    B_3 = B[M//2:,:N//2]
    B_4 = B[M//2:,N//2:]
    return B_1, B_2, B_3, B_4
=======

"""
def submatrices(A):
    
B_upleft = B[:N//2,:M//2]
B_upright = B[N//2:,:M//2]
B_downleft = B[:N//2,M//2:]
B_downright = B[N//2:,M//2]

"""
>>>>>>> origin/master
