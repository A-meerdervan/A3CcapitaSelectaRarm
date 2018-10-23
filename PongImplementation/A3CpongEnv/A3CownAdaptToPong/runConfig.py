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

class RunConfig:
    def __init__(self):
# =======================================================================
# PARS USED in multiple places
        
        # How to display?:
        self.RENDER_SCREEN = False
        self.SHOW_NORMAL_SPEED = False # only if this is true, the framerate has effect.
        self.FPS = 30 # frame rate per second
        
# =======================================================================
# PARS USED ONLY IN MAIN
        #self.ACTIONS = 3 # Number of Actions.  Acton istelf is a scalar:  0:stay, 1:Up, 2:Down
        #self.STATECOUNT = 6 # Size of State [ PlayerYPos, OpponentYPos, BallXPos, BallYPos, BallXDirection, BallYDirection]
        
        # Run specific parameters
        self.MAX_EP_LENGTH = 10000
        self.GAMMA = .99 # discount rate for advantage estimation and reward discounting
        self.L_RATE = 1e-3 # was at 7e-4 for the Dm-mc implementation
        self.ALPHA = 0.99 # was at 0.99 originally, openAI also had 0.99
        self.EPSILON = 1e-5 # was at 0.1 originally, openAI had 1e-5
        self.LOAD_MODEL = False
        self.OUTP_FOLDER = '/verwijderDITTT'
        self.NUM_WORKERS = 12 #multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        # ENV specific
        self.ACTION_SPACE = [1,2,3]
        self.S_SIZE = 6 # Our pong version has a state of 6 numbers
        self.A_SIZE = 3 # Agent can move up down or do nothing
        
        # Not run specific parameters
        self.OUTP_FOLDER = './LogsOfRuns' + self.OUTP_FOLDER
        self.TF_SUMM_PATH = self.OUTP_FOLDER + '/train_'
        self.MODEL_PATH = self.OUTP_FOLDER + '/model'

# Pars used in NETWORK
        self.N_HID_NODES = 64
        self.VF_COEF = 0.5 # Was originally at 0.5, openAI also had 0.5
        self.ENT_COEF = 0.01 # Was at 0.1 originally, openAI has 0.01 as default in A2C
        self.MAX_GRAD_NORM = 0.5 # was originally 40 and openAI has 0.5 in A2C

# Pars used in WORKER
# ========================================================================
        self.TF_SUM_EP_INTVL = 5 # was 5
        self.TF_SAVE_MODEL_INTVL = 250 # was 250
        self.NSTEPS = 5 # was at 20 for the Dm-mc implementation and 5 is default for openAI A2C        

# ========================================================================
# PONG GAME PARAMETERS
        # This is used in the NormalizeGameState functions and in the SimpleBot.py
        self.NORMALISATION_FACTOR = 400
        # Bot settings
        self.TRAIN_BOT_TYPE = 10
        
        #size of our window
        self.WINDOW_WIDTH = 500
        self.WINDOW_HEIGHT = 380
        
        #size of our paddle
        self.PADDLE_WIDTH = 10
        self.PADDLE_HEIGHT = 60
        #distance from the edge of the window
        self.PADDLE_BUFFER = 15
        
        #size of our ball
        self.BALL_WIDTH = 10
        self.BALL_HEIGHT = 10
        
        #speeds of our paddle and ball
        self.PADDLE_SPEED = 1.5 # 2.5
        self.BALL_X_SPEED = 3
        self.BALL_Y_SPEED = 3
        self.DFT = 7.5
        
        #RGB colors for our paddle and ball
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255,0,0)
        self.BLUE = (0,0,255)
        self.YELLOW = (255,255,0)

#    def setTrainBotType(self, trainBot):
#        self.TRAIN_BOT_TYPE = trainBot
        

rc = RunConfig()

        
