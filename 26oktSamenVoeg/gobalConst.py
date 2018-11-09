# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:28:34 2018

@author: arnol
"""

import numpy as np

# REAL setup parameters
REAL_webcamWindowWidth = 1920
REAL_webcamWindowHeight = 1080
# Dit is afhankelijk van de resolutie!
REAL_markerAreaTreshold = 200 # must the marker area must be larger than this. For 480p the markers start around 60?
REAL_collorOfWalls = (0,0,255) # RGB value (0,0,255)=red
REAL_wallThickness = 5 # in pixels
REAL_goalColor = (0,255,0) # (0,255,0) = green
REAL_comPort = 'COM3'
REAL_webcamIndex = 1
REAL_videoPath = 'firstVid.avi'

ENV_IS_RARM = True

# PARS FOR EVALUATION ONLY
EVAL_MODE = True
REAL_SETUP = False
EVAL_RENDER = True # only relevant ruing evaluation episodes
EVAL_SHOW_NORMAL_SPEED = True # only relevant during evaluation episodes
EVAL_FPS = 120 # only relevant during evalutation episodes
#EVAL_CPU_CNT = 12 # Number of cpu's used during training
EVAL_FOLDER = 'run06novReachable5050G993' # The folder that holds the model and train folders
# deze heb je nodig om iets te evalueren in een compleet andere map
#EVAL_HIGHLEVEL_FOLDER = './LogsOfRuns/TempTransferCluster29okt'
#tussen = EVAL_HIGHLEVEL_FOLDER + '/' + EVAL_FOLDER
#EVAL_MODEL_PATH = tussen + "/model"
EVAL_NR_OF_GAMES = 100
# ===================================================================
# PARS FOR ANGLE/BODY TEST MODE (only works if EVAL_MODE = FALSE !!!)
TEST_MODE = False
TEST_fromTestConsts = True
TEST_U_INPUT = True
#TEST_rob_ResetAngles = np.radians(np.array([130,-92,100])) # This is oriented for going left
TEST_rob_ResetAngles = np.radians(np.array([50,81,-90])) # This is oriented for going left
# Not yet used
#TEST_rob_ResetAngles_Left = np.radians(np.array([130,-92,100])) # This is oriented for going left
#TEST_rob_ResetAngles_Right = np.radians(np.array([50,81,-90])) # this is oriented for going right.


sim_WINDOW_WIDTH = 400
sim_WINDOW_HEIGHT = 400
sim_RandomGoal = True
sim_SparseRewards = False
sim_Goal = np.array([105,272])
sim_AddNoise = True
sim_goalRadius = 5
sim_GoalReward = 100.
sim_expRewardGamma = -0.022 # was 0.01 before 6 nov
sim_expRewardOffset = 100
sim_thresholdWall = 10. # linear punishmet starts at this amount of pixels
sim_WallReward = 100
sim_defaultEnvNr = 1
sim_Y_threshold_goal = 370 - sim_thresholdWall #TODOG
# the normalisation is dependend on whether sparse rewards are used
if sim_SparseRewards:
    # the higest reward must be 1, this ensures that is true.
    sim_rewardNormalisation = max(sim_GoalReward,sim_WallReward)
else: # in case no sparse rewards
    sim_rewardNormalisation = sim_WallReward + sim_expRewardOffset
print("GlobalCOnst ",sim_rewardNormalisation)


rob_RandomInit = True
rob_RandomWalls = True
rob_UseSetupBody = True
rob_NoiseStandDev = 0.001
rob_StepSize = np.radians(1.4) # used in real: 2 # 6 nov trained on 1.4
rob_MaxJointAngle = np.radians(np.array([180,0,100,-100,107,-107])) # these are from the actual setup
#rob_MaxJointAngle = np.radians(np.array([180,0,170,-170,170,-170])) # these are what was used before 5 nov

if rob_UseSetupBody:
    bodyFactor = 0.5 #0.5 after 5nov, Was at 0.9 on 3 nov; used to make the body verhoudingen match our envs.
    rob_JointLenght = bodyFactor*np.array([71,112,141*0.6,141*0.4]) #[100,100,80,20] #100,100,80,20
    rob_JointLenght = rob_JointLenght.astype(int)
    rob_JointWidth = int(bodyFactor*30) # 30 mm wide
else: # this is for the old thin body
    rob_JointLenght = np.array([100,100,80,20]) # this is the original thin body
    rob_JointWidth = 10
#rob_ResetAngles = np.radians(np.array([100,-10,90])) # dont know this one
#rob_ResetAngles = np.radians(np.array([65,115,-115])) # This is the one to the right.
rob_ResetAngles = np.radians(np.array([120,-99,99])) # this is used with randomWalls = True in 31 okt runs
#rob_ResetAngles = np.radians(np.array([130,-140,140])) # this was used during the 29okt runs
rob_resetAngles_Lchance = 0.5 # the chance of having a left oriented init
rob_resetAngles_Rchance = 0.5 # the chance of having a right oriented init
rob_ResetAngles_Left = np.radians(np.array([131,-89,99])) # This is oriented for going left
rob_ResetAngles_Right = np.radians(np.array([48,89,-99])) # this is oriented for going right.





run_Render = True
run_NumOfWorkers = 12
run_MaxEpisodeLenght = 2000
run_FPS = 15
run_Gamma = .99 # discount rate for advantage estimation and reward discounting
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
netw_nHidNodes = 128
netw_vfCoef = 0.5 # Was originally at 0.5, openAI also had 0.5
netw_entCoef = 0.1 # Was at 0.1 originally, openAI has 0.01 as default in A2C
netw_maxGradNorm = 40 # was originally 40 and openAI has 0.5 in A2C


OUTP_FOLDER = '/TestRun'

OUTP_FOLDER = './LogsOfRuns' + OUTP_FOLDER
TF_SUMM_PATH = OUTP_FOLDER + '/train_'
run_ModelPath = OUTP_FOLDER + '/model'


