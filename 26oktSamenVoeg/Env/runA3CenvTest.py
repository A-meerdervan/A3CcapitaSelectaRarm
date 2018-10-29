# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:56:39 2018

@author: Alex
"""
import A3CenvPong
from GlobalConstantsA3C import gc # has high level constants
import SimpleBot
import MyPong

def play1Game(env):
    s = env.reset()
    
#    GameState = MyPong.captureNormalisedStateA3C([200.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    #GameStateOpp = captureNormalisedStateA3C([200.0, 200.0, 200.0, 200.0, 1.0, 1.0])
    
    botType = 10
    theAgent = SimpleBot.SimpleBot(botType)
    done = False
#    s = GameState
    i = -1 # TODO remove, was for debug
    
    while ( not done):

        # Determine Next Action From the Agent
        bestAction = theAgent.Act_Mature(s)
        if gc.RENDER_SCREEN: env.render()
        # Take a step in the env
        s1, r, done, i = env.step(bestAction)

        s = s1 # needed for next timestep
#        # Periodically check if the game window was closed, if so, quit.
#        if i % 200 == 0:        
#            if env.theGame.hasQuit():
#                # Close the pygame window
#                env.theGame.quit_()

def playGames():
    #Create our PongGame instance
    env  = A3CenvPong.A3CenvPong()

    play1Game(env)
    play1Game(env)
    env.quitScreen()
    print("ewa")
        
        
def main():
    playGames()
    
if __name__ == "__main__":
    try:
        main()
#        if gc.RENDER_SCREEN: MyPong.quit_()
    except:
        print("Unexpected error in main()")
        #Quit the pygame window
#        if gc.RENDER_SCREEN: MyPong.quit_()
        raise
    