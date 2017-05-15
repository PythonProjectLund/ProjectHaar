# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:37:49 2017

@author: Simon
"""
from  scipy import *
from  pylab import *
import scipy.misc as sm
import scipy.sparse as sparse

v = sm.imread('kvinna.jpg', True)

#cut away pixels to have even number of y and x pixels
if len(v[:, 0]) % 2 != 0:
    v = v[1:, :]

if len(v[0, :]) % 2 != 0:
    v = v[:, 1:]
    
y = len(v[:, 0])
x = len(v[0, :])


"""
Algoritmen tar W*v = y.
Vi önskar konstruera W. Första gör vi v endimensionell och sedan konstruerar
vi en allmän W från denna längd
"""

v = v.reshape(1, -1)[0]

def createW(n):
	W = zeros([n,n])
	i = 0
	for j in range(int(n/2)):
		W[j, i] = 1/2
		W[j, i+1] = 1/2
		i += 2
	i = 0
	for j in range(int(n/2), n):
		W[j, i] = -1/2
		W[j, i+1] = 1/2
		i += 2
	return W


v = array([100, 200, 44, 50, 20, 20, 4, 2])

W = createW(len(v))

y = dot(W, v)

print(y)
#hejsan
#yo