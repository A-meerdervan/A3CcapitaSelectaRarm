# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:08:14 2018

@author: Alex
"""

import numpy as np

s = [0.0075, 0.39, 0.4875, 0.16, 0., 0., 0., 0., 0., 0., 0., 0., 0.77777778, -0.77777778, 0.77777778, -0.20060168, -0.21084209]
s = np.array(s)
sAngles= s[12:15]
print(sAngles)
# dus de start locatie moet kloppen en de end locatie is wat je wilt +1
#print(s[2:4])
Rangles = sAngles * 3.141592653589793
print(Rangles)
#start angles:
rob_ResetAngles = np.radians(np.array([140,-140,140]))
print(rob_ResetAngles)


