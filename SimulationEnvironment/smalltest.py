# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:47:26 2018

@author: arnol
"""

import numpy as np
#savedAngles = robot.jointAngles # save to restore at the end
maxAngls = np.radians(np.array([180,0,100,-100,107,-107]))

R = np.random.random_sample((3)) # get 3 values from 0 to 1
th = np.zeros(3) # init the angles
for i in range(3):
    # have the angles range from their min and max value
    th[i] = maxAngls[0+2*i] + R[i]*(maxAngls[1+2*i] - maxAngls[0+2*i])

print(np.degrees(th))
#    robot.jointAngles = th
#    [initCorrect, minDtoWall] = self.checkNOCollision()
    # In case the goal is placed within the treshold distance
    # of a wall, than try again for a better one
#    if minDtoWall < 10:
#        initCorrect = False

