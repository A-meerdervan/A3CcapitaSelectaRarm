# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:28:34 2018

@author: arnol
"""

import numpy as np

ENV_IS_RARM = True

# PARS FOR EVALUATION ONLY
EVAL_MODE = True
EVAL_SHOW_NORMAL_SPEED = True # only relevant during evaluation episodes
EVAL_FPS = 30 # only relevant during evalutation episodes
EVAL_RENDER = False # only relevant ruing evaluation episodes
#EVAL_CPU_CNT = 12 # Number of cpu's used during training
EVAL_FOLDER = 'runRandomG96' # The folder that holds the model and train folders
EVAL_NR_OF_GAMES = 100
# ===================================================================

sim_WINDOW_WIDTH = 400
sim_WINDOW_HEIGHT = 400

sim_RandomGoal = True
sim_Goal = np.array([100,200])
sim_AddNoise = True
sim_goalRadius = 5

rob_RandomInit = False
rob_NoiseStandDev = 0.001
rob_StepSize = np.radians(1.4)
rob_MaxJointAngle = np.radians(np.array([170,-170]))
rob_JointLenght = [100, 100, 80, 20]
rob_JointWidth = 10
rob_ResetAngles = np.radians(np.array([130,-140,140]))

run_Render = True
run_NumOfWorkers = 12
run_MaxEpisodeLenght = 400
run_FPS = 15
run_Gamma = .96 # discount rate for advantage estimation and reward discounting
run_sSize = 17 # Observations are greyscale frames of 84 * 84 * 1
run_aSize = 6 # Agent can rotate each joint in 2 directions
run_LoadModel = True
run_LearningRate = 1e-3
run_BufferSize = 30
run_actionSpace = [1,2,3,4,5,6]


OUTP_FOLDER = '/runRandomG96dezeVerwijderen'

OUTP_FOLDER = './LogsOfRuns' + OUTP_FOLDER
TF_SUMM_PATH = OUTP_FOLDER + '/train_'
run_ModelPath = OUTP_FOLDER + '/model'


