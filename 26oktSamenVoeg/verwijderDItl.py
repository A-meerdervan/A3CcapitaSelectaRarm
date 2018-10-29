# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:08:14 2018

@author: Alex
"""

import numpy as np
import gobalConst as cn

#s = [0.0075, 0.39, 0.4875, 0.16, 0., 0., 0., 0., 0., 0., 0., 0., 0.77777778, -0.77777778, 0.77777778, -0.20060168, -0.21084209]
#s = np.array(s)
#sAngles= s[12:15]
#print(sAngles)
## dus de start locatie moet kloppen en de end locatie is wat je wilt +1
##print(s[2:4])
#Rangles = sAngles * 3.141592653589793
#print(Rangles)
##start angles:
#rob_ResetAngles = np.radians(np.array([140,-140,140]))
#print(rob_ResetAngles)

def computeReward(minDistance, d, collision=False):
    # compute distance to goal
    # TODO: put variables in the globalConstants
    
    reachedGoal = False
    reward = 0
    #print('inside CR d= ',d)
    if (d <= cn.sim_goalRadius):
     #   print('goal reached want d= ',d,' kleiner dan of gelijk aan ',cn.sim_goalRadius)
        # reached goal
        reachedGoal = True
        reward = cn.sim_GoalReward
    else:
        reachedGoal = False
    if reachedGoal:
        a = 0
      #  print('terminalReward 06 ',reward)
    # Compute the reward relative to the distance of the end effector
    # to the goal. This is caculated exponentially
    gamma = cn.sim_expRewardGamma # this sets the slope
    offset = cn.sim_expRewardOffset  # this determines the maximum negative reward
    # only calculate the relative punishment if the goal has not been reached
    if not reachedGoal and not cn.sim_SparseRewards:
        reward = reward + offset * np.exp(gamma * d) - offset
    if reachedGoal:
        a = 0
        #print('terminalReward 07 ', reward)
    # Calculate the linear punishment when being near to the wall
    thresholdWall = cn.sim_thresholdWall # the amount of pixels where the linear rewards starts
    wallReward = cn.sim_WallReward
    if collision:
        reward = reward - wallReward
    elif minDistance < thresholdWall:
        reward = reward - (thresholdWall - minDistance) * (wallReward/thresholdWall)

    if reachedGoal:
        a = 0
        #print('terminalReward 08 ',reward)
        #print('terminalReward 09 ',reward/cn.sim_rewardNormalisation)
    return [reward/cn.sim_rewardNormalisation, reachedGoal]
ds = [0,1,2,3,4,5]
minDs = [-5,0,5,10,15,20,25,30,40]
for d in ds:
    for minD in minDs:
        print('d ',d,' minD ',minD)
        print('R: ',computeReward(minD,d)[0] * cn.sim_rewardNormalisation )

d = 5
minD = 30
print('_____________\nd ',d,' minD ',minD)
print('R: ',computeReward(minD,d)[0] * cn.sim_rewardNormalisation )

