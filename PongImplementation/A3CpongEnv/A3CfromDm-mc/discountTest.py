# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:43:01 2018

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import scipy

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

ewa = [1,2,3,4,5]
ew = ewa + [100]
nparr = np.array(ew)
gamma = 0.9

rewDisc = discount(nparr,gamma)
print(rewDisc)

print(ew[0] + ew[1]*np.power(gamma,1) + ew[2]*np.power(gamma,2) + ew[3]*np.power(gamma,3) + ew[4]*np.power(gamma,4) + ew[5]*np.power(gamma,5))



""" 
[1,2,3,4,5] + [100] , the 10 is the bootstrapped value
-> [1,2,3,4,5,100]
then do the discount function and you get:
[70.4755  77.195   83.55    89.5     95.     100.    ]
"""

