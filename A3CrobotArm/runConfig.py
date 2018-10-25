# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:21:31 2018

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:46:29 2018

@author: Alex
"""

import numpy as np

#class RunConfig:
#    def __init__(self):
# =======================================================================
        
ENV_IS_RARM = True
# How to display?:
RENDER_SCREEN = True
SHOW_NORMAL_SPEED = False # Is used in evalModel.py
FPS = 30 # frame rate per second

# =======================================================================
# PARS USED ONLY IN MAIN

# Run specific parameters
MAX_EP_LENGTH = 10000
GAMMA = .99 # discount rate for advantage estimation and reward discounting
L_RATE = 7e-3 # was at 7e-4 for the Dm-mc implementation
ALPHA = 0.99 # was at 0.99 originally, openAI also had 0.99
EPSILON = 1e-5 # was at 0.1 originally, openAI had 1e-5
LOAD_MODEL = False
OUTP_FOLDER = '/verwijderDittttSparse'
NUM_WORKERS = 4 #multiprocessing.cpu_count() # Set workers ot number of available CPU threads
 
if ENV_IS_RARM:
#            # Our Robot SIM
    ACTION_SPACE = [1,2,3,4,5,6]
    S_SIZE = 17 # Our pong version has a state of 6 numbers
    A_SIZE = 6 # Agent can move up down or do nothing
else:
        # PONG state of 6 numbers
    ACTION_SPACE = [1,2,3]
    S_SIZE = 6 # Our pong version has a state of 6 numbers
    A_SIZE = 3 # Agent can move up down or do nothing

# Not run specific parameters
OUTP_FOLDER = './LogsOfRuns' + OUTP_FOLDER
TF_SUMM_PATH = OUTP_FOLDER + '/train_'
MODEL_PATH = OUTP_FOLDER + '/model'

# Pars used in NETWORK
N_HID_NODES = 64
VF_COEF = 0.5 # Was originally at 0.5, openAI also had 0.5
ENT_COEF = 0.1 # Was at 0.1 originally, openAI has 0.01 as default in A2C
MAX_GRAD_NORM = 0.5 # was originally 40 and openAI has 0.5 in A2C

# Pars used in WORKER
# ========================================================================
TF_SUM_EP_INTVL = 20 # was 5
TF_SAVE_MODEL_INTVL = 250 # was 250
NSTEPS = 20 # was at 20 for the Dm-mc implementation and 5 is default for openAI A2C        

# OWN robot arm sim PARAMETERS
# ========================================================================
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
rob_JointAngles = np.radians(np.array([105,-90,120]))

# ========================================================================
# PONG GAME PARAMETERS
# This is used in the NormalizeGameState functions and in the SimpleBot.py
NORMALISATION_FACTOR = 400
# Bot settings
TRAIN_BOT_TYPE = 10

#size of our window
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 380

#size of our paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
#distance from the edge of the window
PADDLE_BUFFER = 15

#size of our ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

#speeds of our paddle and ball
PADDLE_SPEED = 1.5 # 2.5
BALL_X_SPEED = 3
BALL_Y_SPEED = 3
DFT = 7.5

#RGB colors for our paddle and ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)

#rc = RunConfig()

        
