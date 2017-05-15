# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:37:49 2017

@author: Simon
"""
from  scipy import *
from  pylab import *
import scipy.misc as sm
import scipy.sparse as sparse

A = sm.imread('kvinna.jpg', True)
#b1,b2,b3,b4


compress(A)

def compress(A):
    submatrices()

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

#cut away pixels to have even number of y and x pixels
if len(A[:, 0]) % 2 != 0:
    A = A[1:, :]

if len(A[0, :]) % 2 != 0:
    A = A[:, 1:]


#vi sätter upp att A är en MxN matris    
M = len(A[:, 0])
N = len(A[0, :])

#v = array([100, 200, 44, 50, 20, 20, 4, 2])
#W = createW(len(v))

W_M = create_W(M)
W_N = create_W(N)

B = dot(dot(W_M, A), W_N.T)
#B = dot(W_M, A)

print(B)

sm.imsave('komprimerad.jpg', B)



print(B)

def submatrices(B):
    B_1 = B[:M//2,:N//2]
    B_2 = B[:M//2,N//2:]
    B_3 = B[M//2:,:N//2]
    B_4 = B[M//2:,N//2:]
    return B_1, B_2, B_3, B_4
