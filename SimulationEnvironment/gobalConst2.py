# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:28:34 2018

@author: arnol
"""

import numpy as np

sim_WINDOW_WIDTH = 400
sim_WINDOW_HEIGHT = 400

sim_RandomGoal = False
sim_Goal = np.array([100,200])
sim_AddNoise = True

rob_NoiseStandDev = 0.001
rob_StepSize = np.radians(1.4)
rob_MaxJointAngle = np.radians(np.array([170,-170]))
rob_JointLenght = [100, 100, 80, 20]
rob_JointWidth = 10
rob_ResetAngles = np.radians(np.array([120,-140,140]))



run_Render = True
run_FPS = 15
run_MaxEpisodeLenght = 10000
run_Gamma = .95 # discount rate for advantage estimation and reward discounting
run_sSize = 17 # Observations are greyscale frames of 84 * 84 * 1
run_aSize = 6 # Agent can rotate each joint in 2 directions
run_LoadModel = False
run_ModelPath = './model'
run_LearningRate = 1e-3
run_BufferSize = 30