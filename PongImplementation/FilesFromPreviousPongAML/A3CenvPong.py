# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:21:46 2018

@author: Alex
"""
import MyPong
import SimpleBot # needed for the opponent in the env.
from GlobalConstants import gc # has high level constants

# =======================================================================
# Normalise GameState
def captureNormalisedStateA3C(gState):
    gState[0] = gState[0]/gc.NORMALISATION_FACTOR    # Normalised PlayerYPos
    gState[1] = gState[1]/gc.NORMALISATION_FACTOR       # Normalised OpponentYpos
    gState[2] = gState[2]/gc.NORMALISATION_FACTOR    # Normalised BallXPos
    gState[3] = gState[3]/gc.NORMALISATION_FACTOR    # Normalised BallYPos
    gState[4] = gState[4]/1.0    # Normalised BallXDirection
    gState[5] = gState[5]/1.0    # Normalised BallYDirection

    return gState

class A3CenvPong:
    def __init__(self):
        self.botType = 10 # this is the speed of the bot paddle 10 is standard
        self.draw = True
        # TODO miss de draw uit de init van PongGame halen
        self.theGame = MyPong.PongGame(self.botType, self.draw)
        self.oppAgent = SimpleBot.SimpleBot(self.botType) 
        self.screenActive = False
        
    def reset(self):
        self.theGame.resetEpisode()
        
    def render(self):
        if (not self.screenActive):
            self.theGame.createWindow()
        self.theGame.renderCurrentState()
    # This function takes the action chosen by the agent and moves the environment
    # 1 timestep to the future and returns the new reward, state and such
    def step(self, actionNr):
        # Determine next action from opponent
        gameStateOpp = captureNormalisedStateA3C(self.theGame.returnCurrentStateOpp())
        oppAction = self.oppAgent.Act_Mature(gameStateOpp)
        # This function returns the new state and the obtained reward but not yet normalised
        s1,r,d,i = self.theGame.PlayNextMoveA3C(actionNr, oppAction)
        # Normalize the state:
        s1 = captureNormalisedStateA3C(s1)
        # Return: The next state, the imidiate reward, done = episode terminated?, i = the timestep
        return s1,r,d,i

def play1Game(env):
    env.reset()
    GameState = captureNormalisedStateA3C([200.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    #GameStateOpp = captureNormalisedStateA3C([200.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    
    botType = 10
    theAgent = SimpleBot.SimpleBot(botType)
    done = False
    s1 = GameState
    
    while ( not done):

        # Determine Next Action From the Agent
        bestAction = theAgent.Act_Mature(s1)
        
        if gc.RENDER_SCREEN: env.render()
        # Take a step in the env
        s1, r, done, i = env.step(bestAction)

        # Periodically check if the game window was closed, if so, quit.
        if i % 200 == 0:        
            if env.theGame.hasQuit():
                # Close the pygame window
                env.theGame.quit_()

def playGames():
    #Create our PongGame instance
    env  = A3CenvPong()

    play1Game(env)
    play1Game(env)
    print("ewa")
        
        
def main():
    playGames()
    
if __name__ == "__main__":
    try:
        main()
        if gc.RENDER_SCREEN: MyPong.quit_()
    except:
        print("Unexpected error")
        #Quit the pygame window
        if gc.RENDER_SCREEN: MyPong.quit_()
        raise
    
    