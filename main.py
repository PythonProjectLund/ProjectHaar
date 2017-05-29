# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:37:49 2017

@author: Simon
"""
from  scipy import *
from  pylab import *
import scipy.misc as sm
import numpy as np
import time

def create_W(n):
    W = zeros([n,n])
    i = 0
    scale_factor = 1/sqrt(2) 
    
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
    return (compressmore(compress_Albinstyle(A)))
    
def compress_Albinstyle(A):
    A, M, N = preprocess_matrix(A)
    B_m = zeros([M,N])
    for x in range(M):
        for y in range(N//2):
            B_m[x,y]=(A[x,2*y-1]+A[x,2*y])/2
        for y in range(N//2):
            B_m[x,y+(N//2)]=(A[x,2*y]-A[x,2*y-1])/2
    
    return B_m

def compressmore(A):
    A, M, N = preprocess_matrix(A)
    B_m = zeros([M,N])
    for y in range(N):
        for x in range(M//2):
            B_m[x,y]=(A[2*x-1,y]+A[2*x,y])/2
        for x in range(M//2):
            B_m[x+(M//2),y]=(A[2*x,y]-A[2*x-1,y])/2
    
    return B_m

def merge_submatrices(B1, B2, B3, B4):
    Q1 = np.column_stack([B1, B2])
    Q2 = np.column_stack([B3, B4])
    
    return np.vstack([Q1, Q2])

def inverse_transformation_piecewise(B1, B2, B3, B4):
    return inverse_transformation(merge_submatrices(B1, B2, B3, B4))

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
    Takes an array and compresses it "levels" times, discarding deltas
    """
    for i in range(levels):
        old = submatrices(compress(old))[0]
    
    return old

def lossy_submatrix_compression(A, threshold):
    A1, A2, A3, A4 = submatrices(A)
    
    M = len(A2[:, 0])
    N = len(A2[0, :])
    
    x = 0
    y = 0
    
    o = 0
    p = 0
    
    for i in range(M):
        for j in range(N):
            #no pixels are zero, those who represent black are something
            #along the lines of 1.e-16 so we count how many of those there
            #are to accurately measure what effect the filter has
            if abs(A2[i, j]) < 0.000001 and A2[i, j] != 0:
                o += 1
            else:
                p += 1
            if abs(A2[i, j]) <= threshold:
                A2[i, j] = 0
                x += 1
            else:
                y += 1
            if abs(A3[i, j]) <= threshold:
                A3[i, j] = 0
                x += 1
            else:
                y += 1
            if abs(A4[i, j]) <= threshold:
                A4[i, j] = 0
                x += 1
            else:
                y += 1
            
    print(o)
    print(p)
    print(x)
    print(y)
    
    return A1, A2, A3, A4

A = compress(sm.imread('kvinna.jpg', True))

B1, B2, B3, B4 = lossy_submatrix_compression(A, 20)
B = merge_submatrices(B1, B2, B3, B4)

sm.imsave('xxx.jpg', B)
sm.imsave('yyy.jpg', inverse_transformation(B))



#Random and dark
ran = rand(500, 500)
sm.imsave('random.jpg', ran)
sm.imsave('dark.jpg',   zeros([500, 500]))
crand = compress(ran)
sm.imsave('random_comp.jpg', crand)
crand2 = compress(crand)
sm.imsave('random_comp2.jpg', compress(crand2))
sm.imsave('random_a_res.jpg', inverse_transformation(inverse_transformation(crand2)))



A0 = sm.imread('gruppen.jpg', True)
A1 = compress(A0)
sm.imsave('A1.jpg', A1)
A2 = compress(A1)
sm.imsave('A2.jpg', A2)
A3 = compress(A2)
sm.imsave('A3.jpg', A3)
A4 = compress(A3)
sm.imsave('A4.jpg', A4)
A5 = compress(A4)
sm.imsave('A5.jpg', A5)
A6 = compress(A5)
sm.imsave('A6.jpg', A6)
A7 = compress(A6)
sm.imsave('A7.jpg', A7)
A8 = compress(A7)
sm.imsave('A8.jpg', A8)

"""
PLOTTING

xs = []
ys = []

xs2 = []
ys2 = []
for n in range(1, 100):
    ran = []
    for i in range(30):
        ran.append(rand(n*10, n*10))
    
    res = []
    
    time1 = time.time()
    for i in range(30):
        res.append(compress(ran[i]))
    time2 = time.time()
    #ys.append((time2 - time1)*1000/40)
    #xs.append(n*10)

    time3 = time.time()
    for i in range(30):
        inverse_transformation(res[i])
    time4 = time.time()
    ys.append((time4 - time3)*1000/30)
    xs.append(n*10)   



plot(xs, ys)
title('Decompression (matrix)')
xlabel('Image size')
ylabel('Compress time (ms)')

"""


time1 = time.time()
B = compress(A)
time2 = time.time()

time3 = time.time()
B_nomat = compress_no_matrices(A)
time4 = time.time()

print('Matrix multiplication: {}'.format(time2 - time1))
print('No matrices: {}'.format(time4 - time3))
print('Time ration: {}'.format((time4 - time3) / (time2 - time1)))

sm.imsave('compressed_nomat.jpg', B_nomat)
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

B1, B2, B3, B4 = lossy_submatrix_compression(compress(sm.imread('kvinna.jpg', True)), 50)
sm.imsave('kvinna_lossy_noinverse.jpg', merge_submatrices(B1, B2, B3, B4))
sm.imsave('kvinna_lossy.jpg', inverse_transformation_piecewise(B1, B2, B3, B4))

sm.imsave('gruppen_lossless.jpg', inverse_transformation(compress(A)))


B = compress_levels(A, 2)
B_temp = compress(B)
B1, B2, B3, B4 = lossy_submatrix_compression(B_temp, 1)

C = inverse_transformation_piecewise(B1, B2, B3, B4)
sm.imsave('gruppen_downscaled_lossy.jpg', C)
sm.imsave('gruppen_downscaled_lossy_compressed.jpg', compress(C))


