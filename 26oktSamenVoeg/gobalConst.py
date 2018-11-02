# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:28:34 2018

@author: arnol
"""

import numpy as np

ENV_IS_RARM = True

# PARS FOR EVALUATION ONLY
EVAL_MODE = True
EVAL_RENDER = True # only relevant ruing evaluation episodes
EVAL_SHOW_NORMAL_SPEED = True # only relevant during evaluation episodes
EVAL_FPS = 60 # only relevant during evalutation episodes
#EVAL_CPU_CNT = 12 # Number of cpu's used during training
EVAL_FOLDER = 'run31oktNachtG98ookEnv7' # The folder that holds the model and train folders
# deze heb je nodig om iets te evalueren in een compleet andere map
#EVAL_HIGHLEVEL_FOLDER = './LogsOfRuns/TempTransferCluster29okt'
#tussen = EVAL_HIGHLEVEL_FOLDER + '/' + EVAL_FOLDER
#EVAL_MODEL_PATH = tussen + "/model"
EVAL_NR_OF_GAMES = 100
# ===================================================================

sim_WINDOW_WIDTH = 400
sim_WINDOW_HEIGHT = 400

sim_RandomGoal = True
sim_SparseRewards = False
sim_Goal = np.array([100,200])
sim_AddNoise = True
sim_goalRadius = 5
sim_GoalReward = 100.
sim_expRewardGamma = -0.01
sim_expRewardOffset = 100
sim_thresholdWall = 20. # linear punishmet starts at this amount of pixels
sim_WallReward = 100
# the normalisation is dependend on whether sparse rewards are used
if sim_SparseRewards:
    # the higest reward must be 1, this ensures that is true.
    sim_rewardNormalisation = max(sim_GoalReward,sim_WallReward)
else: # in case no sparse rewards
    sim_rewardNormalisation = sim_WallReward + sim_expRewardOffset
print("GlobalCOnst ",sim_rewardNormalisation)


rob_RandomInit = False
rob_RandomWalls = True
rob_NoiseStandDev = 0.001
rob_StepSize = np.radians(1.4)
rob_MaxJointAngle = np.radians(np.array([170,-170]))
rob_JointLenght = [100,100,80,20]#[10, 10, 5, 5] # 100,100,80,20
rob_JointWidth = 10
#rob_ResetAngles = np.radians(np.array([65,115,-115]))
rob_ResetAngles = np.radians(np.array([115,-95,115])) # this is used with randomWalls = True in 31 okt runs
#rob_ResetAngles = np.radians(np.array([130,-140,140])) # this was used during the 29okt runs

run_Render = False
run_NumOfWorkers = 12
run_MaxEpisodeLenght = 500
run_FPS = 15
run_Gamma = .98 # discount rate for advantage estimation and reward discounting
run_sSize = 17 # Observations are greyscale frames of 84 * 84 * 1
run_aSize = 6 # Agent can rotate each joint in 2 directions
run_LoadModel = False
run_LearningRate = 1e-3
run_epsilon = 0.1
run_decay = 0.99
run_BufferSize = 30
run_actionSpace = [1,2,3,4,5,6]
run_TFsummIntrvl = 5 # after this many episodes a datapoint is saved
run_TFmodelSaveIntrvl = 100 # after this many episodes the model is saved.

# Pars used in NETWORK
netw_nHidNodes = 512
netw_vfCoef = 0.5 # Was originally at 0.5, openAI also had 0.5
netw_entCoef = 0.1 # Was at 0.1 originally, openAI has 0.01 as default in A2C
netw_maxGradNorm = 40 # was originally 40 and openAI has 0.5 in A2C


OUTP_FOLDER = '/runLocal31oktNachtG98'

OUTP_FOLDER = './LogsOfRuns' + OUTP_FOLDER
TF_SUMM_PATH = OUTP_FOLDER + '/train_'
run_ModelPath = OUTP_FOLDER + '/model'


