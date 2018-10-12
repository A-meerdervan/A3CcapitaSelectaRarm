# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:43:11 2018

@author: Alex
"""

import MyPong # My PyGame Pong Game
#import MyAgent # My DQN Based Agent
#import SimpleBot
#import SelfplayAgent # previous version of DQN agent
import numpy as np
#import random
#import matplotlib.pyplot as plt
from GlobalConstants import gc

# SEE GLOBALCONSTANTS.py 
# For hyperparameters

# =======================================================================
# Normalise GameState
def CaptureNormalisedState(PlayerYPos, OppYpos, BallXPos, BallYPos, BallXDirection, BallYDirection):
    gstate = np.zeros([gc.STATECOUNT])
    gstate[0] = PlayerYPos/gc.NORMALISATION_FACTOR    # Normalised PlayerYPos
    gstate[1] = OppYpos/gc.NORMALISATION_FACTOR       # Normalised OpponentYpos
    gstate[2] = BallXPos/gc.NORMALISATION_FACTOR    # Normalised BallXPos
    gstate[3] = BallYPos/gc.NORMALISATION_FACTOR    # Normalised BallYPos
    gstate[4] = BallXDirection/1.0    # Normalised BallXDirection
    gstate[5] = BallYDirection/1.0    # Normalised BallYDirection

    return gstate
# =======================================================================

def PlayGame(TheAgent, OpponentAgent, renderScreen, saveCSV, csvPath): # ,iteration):
    GameTime = 0
    #Create our PongGame instance
    TheGame = MyPong.PongGame(OpponentAgent.type)
    # Initialise Game
    if renderScreen: TheGame.InitialDisplay()

    GameState = CaptureNormalisedState(200.0, 200.0, 200.0, 200.0, 1.0, 1.0)
    GameStateOpp = CaptureNormalisedState(200.0, 200.0, 200.0, 200.0, 1.0, 1.0)
    
    #reset scores and game stats
    MyPong.gameStats.resetVariables()

    [scoreAI, scoreOld] = MyPong.gameStats.getScore()
    while (scoreAI < gc.GAME_POINTS_WIN and scoreOld < gc.GAME_POINTS_WIN):
        
        # Initialise NextAction  Assume Action is scalar:  0:stay, 1:Up, 2:Down
        BestAction = 0

        # Determine next action from opponent
        OpponentAction = OpponentAgent.Act_Mature(GameStateOpp)
        # Determine Next Action From the Agent
        BestAction = TheAgent.Act_Mature(GameState)
        
        # TODO dit terug veranderen.
        [ReturnScore,PlayerYPos, OppYPos, BallXPos, BallYPos, BallXDirection, BallYDirection]= TheGame.PlayNextMove(BestAction, OpponentAction, GameTime)

        GameStateOpp = CaptureNormalisedState(OppYPos, PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection)
        GameState = CaptureNormalisedState(PlayerYPos, OppYPos, BallXPos, BallYPos, BallXDirection, BallYDirection)

        # update score
        [scoreAI, scoreOld] = MyPong.gameStats.getScore()

        # Move GameTime Click
        GameTime = GameTime+1 
        if GameTime % 200 == 0:        
            #print("succes")
            # Check if the game window was closed, if so, quit.
            if TheGame.hasQuit():
                # Close the pygame window
                TheGame.quit_()
        
    # Save game statistics
    MyPong.gameStats.addGameTime(GameTime)
    # TODO dit verwijderen als het zonder goed blijkt te werken
    # Only save if iteration > 0 , in the TestAgent file, -1 is used for the iteration
    #if (iteration >= 0):
    #    MyPong.gameStats.saveStatsIteration(gc.CSV_FILE_PATH)
    if saveCSV: MyPong.gameStats.saveStatsIteration(csvPath)
    
    if (scoreAI == gc.GAME_POINTS_WIN):
        return [scoreAI,scoreOld,1]
    else:
        return [scoreAI,scoreOld,0]

def TrainAgent(TheAgent, OpponentAgent, renderScreen, iteration):
    
    # This resets the agent to the start of the training proces (epsilon at MAX)
    TheAgent.reset()
    GameTime = 0
    GameHistory = []

    #Create our PongGame instance
    TheGame = MyPong.PongGame(OpponentAgent.type)
    # Initialise Game
    #TODO deze regel weer activeren: 
    if renderScreen: TheGame.InitialDisplay()

    # Initialise NextAction  Assume Action is scalar:  0:stay, 1:Up, 2:Down
    BestAction = 0

    # Initialise current Game State ~ Believe insigificant: (PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection)
    GameState = CaptureNormalisedState(200.0, 200.0, 200.0, 200.0, 1.0, 1.0)
    GameStateOpp = CaptureNormalisedState(200.0, 200.0, 200.0, 200.0, 1.0, 1.0)

     # =================================================================
    # Main training loop
    for gtime in range(gc.TRAIN_TIME):

        # First just Update the Game Display
        if GameTime % 100 == 0:
            TheGame.UpdateGameDisplay(GameTime,TheAgent.epsilon)

        # Determine Next Action From the Agents
        BestAction = TheAgent.Act_Train(GameState)
        OpponentAction = OpponentAgent.Act_Mature(GameStateOpp)
        
        # Run the game dynamics one step
        [ReturnScore,PlayerYPos, OppYPos, BallXPos, BallYPos, BallXDirection, BallYDirection]= TheGame.PlayNextMove(BestAction, OpponentAction, GameTime)
        
        # Capture resulting states
        GameStateOpp = CaptureNormalisedState(OppYPos, PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection)
        NextState = CaptureNormalisedState(PlayerYPos, OppYPos, BallXPos, BallYPos, BallXDirection, BallYDirection)

        # Capture the Sample [S, A, R, S"] in Agent Experience Replay Memory
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState))

        " TODO If both are learning here the captureSample line could be of"
        " the opponent. But it would need it's own NextState line!"

        #  Now Request Agent to DQN Train process  Against Experience
        if GameTime % gc.TRAIN_UPDATE_INTERVAL == 0:
            TheAgent.Process()

        # Move State On
        GameState = NextState

        # Move GameTime Click
        GameTime = GameTime+1

        if GameTime % 200 == 0:
            #print("Game Time: ", GameTime,"  Game Score: ", "{0:.2f}".format(TheGame.GScore), "   EPSILON: ", "{0:.4f}".format(TheAgent.epsilon))
            GameHistory.append((GameTime,TheGame.GScore,TheAgent.epsilon))
            # This makes it possible to press the closing x on the pygame window
            if TheGame.hasQuit():
                # Close the pygame window
                TheGame.quit_() 
