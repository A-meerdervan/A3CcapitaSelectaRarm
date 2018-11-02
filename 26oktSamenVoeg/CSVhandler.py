# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:54:27 2018

@author: Alex
"""

import numpy as np

def saveListToCSV(filepath, _list):
    with open(filepath,'ab') as f:
        np.savetxt(f, [_list], delimiter=',', fmt='%f')