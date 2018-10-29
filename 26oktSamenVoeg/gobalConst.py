# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:28:34 2018

@author: arnol
"""

import numpy as np

ENV_IS_RARM = True

# PARS FOR EVALUATION ONLY
EVAL_MODE = False
EVAL_SHOW_NORMAL_SPEED = True # only relevant during evaluation episodes
EVAL_FPS = 30 # only relevant during evalutation episodes
EVAL_RENDER = False # only relevant ruing evaluation episodes
#EVAL_CPU_CNT = 12 # Number of cpu's used during training
EVAL_FOLDER = 'run29OktSparse99' # The folder that holds the model and train folders
# deze heb je nodig om iets te evalueren in een compleet andere map
#EVAL_HIGHLEVEL_FOLDER = './LogsOfRuns/TempTransferCluster29okt'
#tussen = EVAL_HIGHLEVEL_FOLDER + '/' + EVAL_FOLDER
#EVAL_MODEL_PATH = tussen + "/model"
EVAL_NR_OF_GAMES = 10
# ===================================================================

sim_WINDOW_WIDTH = 400
sim_WINDOW_HEIGHT = 400

sim_RandomGoal = True
sim_SparseRewards = False
sim_Goal = np.array([100,200])
sim_AddNoise = True
sim_goalRadius = 5
sim_GoalReward = 100
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
rob_NoiseStandDev = 0.001
rob_StepSize = np.radians(1.4)
rob_MaxJointAngle = np.radians(np.array([170,-170]))
rob_JointLenght = [100, 100, 80, 20]
rob_JointWidth = 10
rob_ResetAngles = np.radians(np.array([130,-140,140]))

run_Render = True
run_NumOfWorkers = 12
run_MaxEpisodeLenght = 600
run_FPS = 15
run_Gamma = .99 # discount rate for advantage estimation and reward discounting
run_sSize = 17 # Observations are greyscale frames of 84 * 84 * 1
run_aSize = 6 # Agent can rotate each joint in 2 directions
run_LoadModel = False
run_LearningRate = 1e-3
run_BufferSize = 30
run_actionSpace = [1,2,3,4,5,6]
run_TFsummIntrvl = 1 # after this many episodes a datapoint is saved
run_TFmodelSaveIntrvl = 250 # after this many episodes the model is saved.


OUTP_FOLDER = '/verwijderDeze3'

OUTP_FOLDER = './LogsOfRuns' + OUTP_FOLDER
TF_SUMM_PATH = OUTP_FOLDER + '/train_'
run_ModelPath = OUTP_FOLDER + '/model'


