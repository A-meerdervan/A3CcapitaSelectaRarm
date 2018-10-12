"""
Created on Wed Oct  3 15:12:43 2018

@author: Alex
"""
import MyPong # My PyGame Pong Game
#import MyAgent # My DQN Based Agent
import SimpleBot
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
# =========================================================================

def PlayGame(TheAgent, OpponentAgent, renderScreen):
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
        # Periodically check if the game window was closed, if so, quit.
        if GameTime % 200 == 0:        
            if TheGame.hasQuit():
                # Close the pygame window
                TheGame.quit_()
        
    # Save game statistics
    MyPong.gameStats.addGameTime(GameTime)
    # TODO dit verwijderen als het zonder goed blijkt te werken
    # Only save if iteration > 0 , in the TestAgent file, -1 is used for the iteration
    #if (iteration >= 0):
    #    MyPong.gameStats.saveStatsIteration(gc.CSV_FILE_PATH)
    
    if (scoreAI == gc.GAME_POINTS_WIN):
        return [scoreAI,scoreOld,1]
    else:
        return [scoreAI,scoreOld,0]
    
def main():
    renderScreen = True
    botType = 10
    #Play two bots against eachother
    oppAgent = SimpleBot.SimpleBot(botType)
    theAgent = oppAgent
    PlayGame(theAgent, oppAgent, renderScreen)
    
if __name__ == "__main__":
    try:
        main()
        if gc.RENDER_SCREEN: MyPong.quit_()
    except:
        print("Unexpected error")
        #Quit the pygame window
        if gc.RENDER_SCREEN: MyPong.quit_()
        raise