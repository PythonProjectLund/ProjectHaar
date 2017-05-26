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
    scale_factor = 1/2#1/sqrt(2) 
    
    for j in range(int(n/2)):
        W[j, i] = scale_factor
        W[j, i+1] = scale_factor
        i += 2
    i = 0
    for j in range(int(n/2), n):
        W[j, i] = -scale_factor
        W[j, i+1] = scale_factor
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
    
    return B
    
def compress_no_matrices(A):
    """
    First we iterate vertically and split the
    picture into two parts
    """
    A, M, N = preprocess_matrix(A)
    
    
    """
    for a in range(M):
        for b in range(N//2):
            B[a, b] = (A[a, 2 * b]+A[a,2*b+1])/2
            
        for b in range(N//2):   
            B[a, b + N//2] = (A[a, 2 * b] - A[a, 2 * b+1]) / 2
    
        
    for a in range(N):
        for b in range(M//2):
            C[b,a] = (B[b*2,a] + B[b*2+1,a])/2
        for b in range(M//2):   
            C[b+M//2,a] = (B[b*2,a] - B[b*2+1,a])/2
    """
    
    
    

def inverse_transformation_piecewise(B1, B2, B3, B4):
    Q1 = np.column_stack([B1, B2])
    Q2 = np.column_stack([B3, B4])
    return inverse_transformation(np.vstack([Q1, Q2]))

def inverse_transformation(B):
    """
    Takes a full compressed image B or B in parts B1, B2, B3, B4
    with diff information and returns the original matrix by
    doing the inverse transformation.
    """
    
    M = len(B[:, 0])
    N = len(B[0, :])
        
    W_M = create_W(M)
    W_N = create_W(N)
    
    A = dot(dot(W_M.T, B), W_N)
    
    return A

def submatrices(B):
    """
    Takes a full matrix, returns 4 submatrices for each quadrant
    """
    B, M, N = preprocess_matrix(B)        
    
    B1 = B[:M//2,:N//2]
    B2 = B[:M//2,N//2:]
    B3 = B[M//2:,:N//2]
    B4 = B[M//2:,N//2:]
    
    return B1, B2, B3, B4

def compress_levels(old, levels):
    """
    Takes an array and compresses it "levels" times, discarding delta
    """
    for i in range(levels):
        old = submatrices(compress(old))[0]
    
    return old

def lossy_submatrix_compression(A, threshold):
    A1, A2, A3, A4 = submatrices(A)
    
    M = len(A2[:, 0])
    N = len(A2[0, :])
    
    for i in range(M):
        for j in range(N):
            if abs(A2[i, j]) <= threshold:
                A2[i, j] = 0
            if abs(A3[i, j]) <= threshold:
                A3[i, j] = 0
    return A1, A2, A3, A4



"""
    Kvinna och tillhörande
"""
A = sm.imread('kvinna.jpg', True)
B = compress(A)
#B_nomat = compress_no_matrices(A)
#sm.imsave('compressed_nomat.jpg', B_nomat)
sm.imsave('compressed_full.jpg', B)

A_restored = inverse_transformation(B)
sm.imsave('decompressed_full.jpg', A_restored)

B1, B2, B3, B4 = submatrices(B)


for n in range(1, 5):
    sm.imsave('compressed_{}.jpg'.format(n), compress_levels(A, n))

A_restored_piecewise = inverse_transformation_piecewise(B1, B2, B3, B4)

sm.imsave('decompressed_piecewise.jpg', A_restored_piecewise)





"""
    Gruppen
"""
A = sm.imread('gruppen.jpg', True)
B = compress(A)


B1, B2, B3, B4 = submatrices(B)

M = len(B2[:, 0])
N = len(B2[0, :])

for i in range(M):
    for j in range(N):
        #print(B2[i, j])
        if abs(B2[i, j]) <= 4:
            B2[i, j] = 0
        if abs(B3[i, j]) <= 4:
            B3[i, j] = 0              

sm.imsave('gruppen_lossy.jpg', inverse_transformation_piecewise(B1, B2, B3, B4))

sm.imsave('gruppen_lossless.jpg', inverse_transformation(compress(A)))


#downscaled + lossy:
B = compress_levels(A, 2)
B_temp = compress(B)
B1, B2, B3, B4 = lossy_submatrix_compression(B_temp, 1)

sm.imsave('gruppen_downscaled_lossy.jpg', inverse_transformation_piecewise(B1, B2, B3, B4))


