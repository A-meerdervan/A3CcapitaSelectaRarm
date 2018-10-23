# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:46:29 2018

@author: Alex
"""

import numpy as np
import random, numpy, math
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model

class GlobalConstants:
    def __init__(self):
        # This holds all brains
        self.MODELS = np.array([])
        # How to display?:
        self.RENDER_SCREEN = True
        self.SHOW_NORMAL_SPEED = False # only if this is true, the framerate has effect.
        self.LOAD_MODELS_FROM_FILE = False
        # frame rate per second
        # Experiment Performance Seems rather sensitive to Computer performance (As Ball as rate vs Paddle rate sensitivity)
        self.FPS = 40 # 60
        
        # Bot settings
        self.TRAIN_BOT_TYPE = 10
        
        #Number of random training opponents
        self.N_Of_Train_Oppnts = 1
        # Number of games against random previous versions to test fitnes
        self.N_of_Test_Games = 1
        # Relative file path of the statistics CSV
        self.CSV_FILE_TRAIN = "stats.csv"
        #self.CSV_FILE_TEST = "tests.csv"
        self.CSV_FILE_TEST = "testFile3okt.csv"
        #self.CSV_FILE_TEST = "verwijdersDeze.csv"
        # Relative file path of the model folder
        self.MODEL_FOLDER_PATH = "model"
        # Determines whether training starts with random weights or with the initial model
        self.startFromScratch = True
        self.INITIAL_MODEL_NAME = "initial"
        
        # Variables for during a non-training game
        self.EPSILON_MATURE = 0.01 #TODO back to 0.01
        
        # =======================================================================
        # These are parameters for the TestAgent file
        #self.TEST_MODEL_FOLDER_PATH = "../vrijdag19janNachtRun_TT_15000_NO_3_CYCLES_28/model"
        #self.TEST_MODEL_FOLDER_PATH = "../zaterdag19jan40uurRun_TT_20000_NO_3_CYCLES_50/model"
        #self.TEST_MODEL_FOLDER_PATH = "../maandag23janNachtRun_TT_20000_NO_1_CYCLES_50_AGAINST_BOT/model"
        #self.TEST_MODEL_FOLDER_PATH = "../donderdag25jan_ToevoegingAan19janRun_tot_it_100/model"
        #self.TEST_MODEL_FOLDER_PATH = "../zaterdag27jan_trainEachFrame_TT_20000_NO_1_CYCLES_41_AGAINST_BOT/model"
        self.TEST_MODEL_FOLDER_PATH = "/testPong3okt/model"
        
        self.N_GAMES_NO_TRAIN = 5
        
        # =======================================================================
        #   DQN Algorith Paramaters
        self.MODEL_ARCHITECTURE_TYPE = 1
        self.ACTIONS = 3 # Number of Actions.  Acton istelf is a scalar:  0:stay, 1:Up, 2:Down
        self.STATECOUNT = 6 # Size of State [ PlayerYPos, OpponentYPos, BallXPos, BallYPos, BallXDirection, BallYDirection]
        self.TRAIN_TIME = 100  # Time for the AI to train
        self.GAME_POINTS_WIN = 20   # Points a player needs to get in order to win the game
        self.NUMBER_CYCLES = 50   # The number of train - play cycles for which the AI can evolve
        self.TARGET_Q_NN_UPDATE_INTERVAL = 150
        
        # DQN Reinforcement Learning Algorithm  Hyper Parameters
        # Prime agent:
        self.ExpReplay_CAPACITY = 10000
        self.OBSERVEPERIOD = 1000.        # Period actually start real Training against Experieced Replay Batches
        self.BATCH_SIZE = 128*2
        self.GAMMA = 0.99                # Q Reward Discount Gamma
        self.MAX_EPSILON = 1.0
        self.MIN_EPSILON = 0.05
        self.TRAIN_UPDATE_INTERVAL = 10
        #self.LAMBDA = automatically generated using the observation time and total game time in the agent classes.              
        
        # ========================================================================
        # PONG GAME PARAMETERS
        # This is used in the NormalizeGameState functions and in the SimpleBot.py
        self.NORMALISATION_FACTOR = 400

        # This is for when a training session is so long, that it is nice to save intermidiate model nr's.
        self.MODEL_NR = 10
        self.SAVE_MODEL_INTERVAL = 200000
        
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

    def setNrOfCycles(self, nrOfCycles):
        self.NUMBER_CYCLES = nrOfCycles
    
    def setTrainTime(self, trainTime):
        self.TRAIN_TIME = trainTime
        
    def setModelsFolder(self, modelsFolder):
        self.MODEL_FOLDER_PATH = modelsFolder
        
    def setNofTestGames(self, numOfTestGames):
        self.N_of_Test_Games = numOfTestGames
    
    def setNofTrainOpp(self, numOfTrainOpp):
        self.N_Of_Train_Oppnts = numOfTrainOpp
        
    def setNofTestGamesNoTrain(self, numOfGamesNoTrain):
        self.N_GAMES_NO_TRAIN = numOfGamesNoTrain
        
    def setTrainBotType(self, trainBot):
        self.TRAIN_BOT_TYPE = trainBot
        
    def setModelType(self, modelType):
        self.MODEL_ARCHITECTURE_TYPE = modelType
        
    def setTrainInterval(self, trainInterval):
        self.TRAIN_UPDATE_INTERVAL = trainInterval
        
    def setQ_NN_updateInterval(self, Qinterval):
        self.TARGET_Q_NN_UPDATE_INTERVAL = Qinterval
        
    def resetModels(self):
        self.MODELS = np.array([])

    def incrementModelNr(self):
        self.MODEL_NR += 1
        
    def getModel(self, modelType):
        # 2 hidden layers
        if modelType == 1:
            model = Sequential()    
            # Simple Model with Two Hidden Layers and a Linear Output Layer. The Input layer is simply the State input.
            model.add(Dense(units=64, activation='relu', input_dim= self.STATECOUNT))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=self.ACTIONS, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]
            model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment
            return model        
        # 6 layers deep, 8 wide
        elif modelType == 2:
            model = Sequential()    
            model.add(Dense(units=8, activation='relu', input_dim=self.STATECOUNT))
            model.add(Dense(units=8, activation='relu'))
            model.add(Dense(units=8, activation='relu'))
            model.add(Dense(units=8, activation='relu'))
            model.add(Dense(units=8, activation='relu'))
            model.add(Dense(units=8, activation='relu'))
            model.add(Dense(units=self.ACTIONS, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]
            model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment
            return model
        # 6 layers deep, 16 wide
        elif modelType == 3:
            model = Sequential()    
            model.add(Dense(units=16, activation='relu', input_dim=self.STATECOUNT))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=self.ACTIONS, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]
            model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment        # 6 layers deep, 32 wide
            return model
        # 6 layers deep, 32 wide
        elif modelType == 4:
            model = Sequential()    
            model.add(Dense(units=32, activation='relu', input_dim=self.STATECOUNT))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dense(units=self.ACTIONS, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]
            model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment        
            return model
        # 6 layers deep, 64 wide
        elif modelType == 5:
            model = Sequential()    
            model.add(Dense(units=64, activation='relu', input_dim=self.STATECOUNT))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dense(units=self.ACTIONS, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]
            model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment        
            return model

gc = GlobalConstants()
